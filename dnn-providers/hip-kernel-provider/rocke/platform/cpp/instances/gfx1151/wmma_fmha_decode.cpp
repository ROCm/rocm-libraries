// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_gfx1151_wmma_fmha_decode.c -- C99-API port of
 * kernels/gfx1151/wmma_fmha_decode.py.
 *
 * Byte-identical builder-call sequence vs the Python build_wmma_fmha_decode
 * and build_wmma_fmha_decode_merge: the same kernel attrs, the same
 * parameters in the same order and the same op emission order. Every nested
 * Python expression is hoisted into ordered temporaries (C++ argument
 * evaluation order is unspecified); a call never has more than one argument
 * that emits ops.
 */

#include "rocke/instance_gfx1151_wmma_fmha_decode.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#define DEC_DEFAULT_NAME "rocke_wmma_fmha_decode"
#define DEC_ROWS ROCKE_WMMA_FMHA_DECODE_ROWS
#define DEC_TILE ROCKE_WMMA_FMHA_DECODE_TILE
#define DEC_MERGE_THREADS ROCKE_WMMA_FMHA_DECODE_MERGE_THREADS
#define DEC_MERGE_COLS 4
#define DEC_MERGE_UNROLL 8
#define DEC_MAX_SPLITS 64
#define DEC_FUSED_MAX_SPLITS 4
#define DEC_SPLIT_KEYS 32
#define DEC_V_PAD 4
#define DEC_MAX_LDS_BYTES 65536
#define DEC_LOG2E 1.4426950408889634
#define DEC_LN2 0.6931471805599453
#define DEC_FP8_INT_UNIT 256.0

namespace
{

using Val = rocke_value_t*;
using Vals = std::vector<Val>;

const char* sz(const char* s)
{
    return s != NULL ? s : "";
}

bool eq(const char* a, const char* b)
{
    return a != NULL && strcmp(a, b) == 0;
}

bool dec_bf16(const rocke_wmma_fmha_decode_spec_t* s)
{
    return eq(s->dtype, "bf16");
}

bool dec_fp8(const rocke_wmma_fmha_decode_spec_t* s)
{
    return sz(s->kv_dtype)[0] != '\0';
}

bool dec_packed(const rocke_wmma_fmha_decode_spec_t* s)
{
    return !eq(s->layout, "dense");
}

bool dec_has_bias(const rocke_wmma_fmha_decode_spec_t* s)
{
    return sz(s->bias)[0] != '\0';
}

bool dec_bias_f32(const rocke_wmma_fmha_decode_spec_t* s)
{
    return eq(s->bias, "f32");
}

/* WmmaFmhaDecodeSpec.natural_scores */
bool dec_natural(const rocke_wmma_fmha_decode_spec_t* s)
{
    return dec_has_bias(s) || s->softcap || s->alibi || s->qq_bias;
}

const rocke_type_t* dec_elem(const rocke_wmma_fmha_decode_spec_t* s)
{
    return dec_bf16(s) ? rocke_bf16() : rocke_f16();
}

bool tri_ok(int v)
{
    return v >= -1 && v <= 1;
}

/* WmmaFmhaDecodeSpec.__post_init__ (as a predicate + reason). */
bool dec_post_init_ok(const rocke_wmma_fmha_decode_spec_t* s, char* reason, size_t cap)
{
    char msg[200];
    if(s->head_size != 64 && s->head_size != 128 && s->head_size != 256)
    {
        snprintf(msg,
                 sizeof(msg),
                 "head_size bucket must be one of (64, 128, 256) (got %d)",
                 s->head_size);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(!eq(s->dtype, "fp16") && !eq(s->dtype, "bf16"))
    {
        snprintf(msg, sizeof(msg), "dtype must be fp16 or bf16 (got '%s')", sz(s->dtype));
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(dec_has_bias(s) && !eq(s->bias, "q") && !eq(s->bias, "f32"))
    {
        snprintf(msg, sizeof(msg), "bias must be '', 'q' or 'f32' (got '%s')", s->bias);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(dec_fp8(s) && !eq(s->kv_dtype, "fp8e4m3"))
    {
        snprintf(msg, sizeof(msg), "kv_dtype must be '' or 'fp8e4m3' (got '%s')", s->kv_dtype);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(!eq(s->layout, "dense") && !eq(s->layout, "ragged") && !eq(s->layout, "paged"))
    {
        snprintf(
            msg, sizeof(msg), "layout must be dense, ragged or paged (got '%s')", sz(s->layout));
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    const int ps = s->page_block_size;
    if(eq(s->layout, "paged"))
    {
        if(ps <= 0 || (ps & (ps - 1)) != 0)
        {
            snprintf(msg,
                     sizeof(msg),
                     "paged page_block_size must be a positive power of two (got %d)",
                     ps);
            rocke_spec_set_reason(reason, cap, msg);
            return false;
        }
    }
    else if(ps != 0)
    {
        rocke_spec_set_reason(
            reason, cap, "page_block_size is only meaningful for the paged layout");
        return false;
    }
    if(s->num_waves != 1 && s->num_waves != 2 && s->num_waves != 4 && s->num_waves != 8)
    {
        snprintf(msg, sizeof(msg), "num_waves must be 1, 2, 4 or 8 (got %d)", s->num_waves);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if((s->dv_split != 0 && s->dv_split != 1 && s->dv_split != 2 && s->dv_split != 4)
       || (s->dv_split != 0 && s->head_size / s->dv_split < 32))
    {
        snprintf(
            msg, sizeof(msg), "invalid dv_split %d for head_size %d", s->dv_split, s->head_size);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    const char* qm = sz(s->q_mode);
    if(qm[0] != '\0' && !eq(qm, "reg") && !eq(qm, "lds"))
    {
        snprintf(msg, sizeof(msg), "q_mode must be '', 'reg' or 'lds' (got '%s')", qm);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(s->waves_per_eu < 0)
    {
        snprintf(msg, sizeof(msg), "waves_per_eu must be >= 0 (got %d)", s->waves_per_eu);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    const char* fam = sz(s->wmma_family);
    if(fam[0] != '\0' && !eq(fam, "gfx11") && !eq(fam, "gfx12"))
    {
        snprintf(msg, sizeof(msg), "wmma_family must be '', 'gfx11' or 'gfx12' (got '%s')", fam);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(!eq(s->scheduler_strategy, "auto") && !eq(s->scheduler_strategy, "default")
       && !rocke_scheduler_strategy_is_valid(s->scheduler_strategy))
    {
        snprintf(
            msg, sizeof(msg), "unsupported scheduler_strategy '%s'", sz(s->scheduler_strategy));
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(!tri_ok(s->v_store_early) || !tri_ok(s->prefetch) || !tri_ok(s->lockstep)
       || !tri_ok(s->fp8_cvt) || !tri_ok(s->tanh_fast))
    {
        rocke_spec_set_reason(reason, cap, "tri-state knobs must be -1 (auto), 0 or 1");
        return false;
    }
    return true;
}

/* _layout(arch, dtype): the dtype's wave32 WMMA atom; false (reason set) when
 * the arch has none. */
bool dec_layout(const char* arch,
                const rocke_wmma_fmha_decode_spec_t* s,
                std::string* op_id,
                bool* gfx12,
                char* reason,
                size_t cap)
{
    char msg[200];
    const rocke_archtarget_t* target = arch != NULL ? rocke_archtarget_from_gfx(arch) : NULL;
    if(target == NULL)
    {
        snprintf(msg, sizeof(msg), "unknown gfx target '%s'", sz(arch));
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    const char* elem = dec_bf16(s) ? "bf16" : "f16";
    if(target->wave_size == 32)
    {
        const std::string g12 = std::string("wmma_gfx12_f32_16x16x16_") + elem;
        if(rocke_archtarget_by_op_id(target, g12.c_str()) != NULL)
        {
            *op_id = g12;
            *gfx12 = true;
            return true;
        }
        const std::string g11 = std::string("wmma_f32_16x16x16_") + elem;
        if(rocke_archtarget_by_op_id(target, g11.c_str()) != NULL)
        {
            *op_id = g11;
            *gfx12 = false;
            return true;
        }
    }
    snprintf(msg, sizeof(msg), "%s has no wave32 RDNA WMMA %s atom", arch, elem);
    rocke_spec_set_reason(reason, cap, msg);
    return false;
}

/* resolve_wmma_fmha_decode_spec (after __post_init__). */
bool dec_resolve(const rocke_wmma_fmha_decode_spec_t* in,
                 const char* arch,
                 rocke_wmma_fmha_decode_spec_t* out,
                 std::string* op_id,
                 bool* gfx12_out,
                 char* reason,
                 size_t cap)
{
    char msg[200];
    if(!dec_post_init_ok(in, reason, cap))
        return false;
    bool gfx12 = false;
    std::string op;
    if(!dec_layout(arch, in, &op, &gfx12, reason, cap))
        return false;
    const char* family = gfx12 ? "gfx12" : "gfx11";
    if(sz(in->wmma_family)[0] != '\0' && !eq(in->wmma_family, family))
    {
        snprintf(msg,
                 sizeof(msg),
                 "spec resolved for %s cannot build for %s (%s)",
                 in->wmma_family,
                 arch,
                 family);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    const bool d256 = in->head_size > 128;
    const bool fp8 = dec_fp8(in);
    int a_dvs, a_ve, a_pf, a_ls;
    const char* a_q;
    const char* a_sched;
    if(gfx12)
    {
        a_dvs = d256 ? 4 : 1;
        a_q = "reg";
        a_ve = 1;
        a_pf = in->head_size != 128 ? 1 : 0;
        a_ls = 1;
        a_sched = d256 ? NULL : "max-ilp";
    }
    else
    {
        a_dvs = d256 ? 2 : 1;
        a_q = d256 ? "lds" : "reg";
        a_ve = 0;
        a_pf = (fp8 && !d256) ? 1 : 0;
        a_ls = 0;
        a_sched = "max-ilp";
    }
    rocke_wmma_fmha_decode_spec_t r = *in;
    r.dv_split = in->dv_split != 0 ? in->dv_split : a_dvs;
    r.q_mode = sz(in->q_mode)[0] != '\0' ? in->q_mode : a_q;
    r.v_store_early = in->v_store_early < 0 ? a_ve : in->v_store_early;
    r.prefetch = in->prefetch < 0 ? a_pf : in->prefetch;
    r.lockstep = in->lockstep < 0 ? a_ls : in->lockstep;
    r.fp8_cvt = in->fp8_cvt < 0 ? ((gfx12 && in->head_size != 128) ? 1 : 0)
                                : ((in->fp8_cvt && gfx12) ? 1 : 0);
    r.tanh_fast = in->tanh_fast < 0 ? ((gfx12 || !d256 || fp8) ? 1 : 0) : in->tanh_fast;
    if(eq(in->scheduler_strategy, "auto"))
        r.scheduler_strategy = a_sched;
    else if(eq(in->scheduler_strategy, "default"))
        r.scheduler_strategy = NULL;
    r.wmma_family = family;
    *out = r;
    if(op_id != NULL)
        *op_id = op;
    if(gfx12_out != NULL)
        *gfx12_out = gfx12;
    return true;
}

/* _lds_dwords(spec) of a resolved spec: per-wave region, Q staging. */
void dec_lds_dwords(const rocke_wmma_fmha_decode_spec_t* s, int* region, int* q)
{
    const int dvt = s->head_size / s->dv_split;
    const int rs = DEC_TILE / 2 + DEC_V_PAD;
    const int rec = 64 + (dvt / 16) * 256;
    *region = dvt * rs > rec ? dvt * rs : rec;
    *q = eq(s->q_mode, "lds") ? DEC_ROWS * (s->head_size / 2 + 4) : 0;
}

int dec_lds_bytes_resolved(const rocke_wmma_fmha_decode_spec_t* s)
{
    int region = 0, q = 0;
    dec_lds_dwords(s, &region, &q);
    return 4 * (s->num_waves * region + q);
}

/* Resolve + LDS budget: is_valid_spec. */
bool dec_valid(const rocke_wmma_fmha_decode_spec_t* spec,
               const char* arch,
               rocke_wmma_fmha_decode_spec_t* resolved,
               std::string* op_id,
               bool* gfx12,
               char* reason,
               size_t cap)
{
    char msg[200];
    if(spec == NULL || arch == NULL)
    {
        rocke_spec_set_reason(reason, cap, "null spec/arch");
        return false;
    }
    if(!dec_resolve(spec, arch, resolved, op_id, gfx12, reason, cap))
        return false;
    const int lds = dec_lds_bytes_resolved(resolved);
    if(lds > DEC_MAX_LDS_BYTES)
    {
        snprintf(msg, sizeof(msg), "LDS %d bytes exceed %d", lds, DEC_MAX_LDS_BYTES);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    return true;
}

rocke_param_opts_t
    dec_opts(bool noalias, int ro_wo /* 0 none, 1 readonly, 2 writeonly */, int align)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.noalias = noalias;
    o.noalias_set = noalias;
    if(ro_wo == 1)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    else if(ro_wo == 2)
    {
        o.writeonly = true;
        o.writeonly_set = true;
    }
    o.align = align;
    o.align_set = true;
    return o;
}

/* _declare_params: identical names, types, attrs and order. */
void dec_declare_params(rocke_ir_builder_t* b, const rocke_wmma_fmha_decode_spec_t* s)
{
    const rocke_type_t* elem = dec_elem(s);
    const bool fp8 = dec_fp8(s);
    const rocke_type_t* kv_elem = fp8 ? rocke_fp8e4m3() : elem;
    const int kv_align = fp8 ? 1 : 16;
    rocke_param_opts_t o;
    o = dec_opts(true, 1, 16);
    (void)rocke_b_param(b, "Q", rocke_ptr_type(b, elem, "global"), &o);
    o = dec_opts(true, 1, kv_align);
    (void)rocke_b_param(b, "K", rocke_ptr_type(b, kv_elem, "global"), &o);
    (void)rocke_b_param(b, "V", rocke_ptr_type(b, kv_elem, "global"), &o);
    o = dec_opts(true, 2, 16);
    (void)rocke_b_param(b, "O", rocke_ptr_type(b, elem, "global"), &o);
    o = dec_opts(true, 2, 4);
    (void)rocke_b_param(b, "LSE", rocke_ptr_type(b, rocke_f32(), "global"), &o);
    o = dec_opts(true, 0, 16);
    (void)rocke_b_param(b, "WS_O", rocke_ptr_type(b, rocke_f32(), "global"), &o);
    o = dec_opts(true, 0, 4);
    (void)rocke_b_param(b, "WS_LSE", rocke_ptr_type(b, rocke_f32(), "global"), &o);
    (void)rocke_b_param(b, "WS_CNT", rocke_ptr_type(b, rocke_i32(), "global"), &o);
    (void)rocke_b_param(b, "scale_log2", rocke_f32(), NULL);
    const char* scalars[] = {"seqlen_q",
                             "seqlen_k",
                             "num_query_heads",
                             "num_kv_heads",
                             "num_v_heads",
                             "head_dim_q",
                             "head_dim_v",
                             "bottom_right",
                             "window_left",
                             "window_right",
                             "write_lse",
                             "num_splits"};
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
    auto in_ptr = [&](const char* name, const rocke_type_t* pointee, int align) {
        rocke_param_opts_t po = dec_opts(true, 1, align);
        (void)rocke_b_param(b, name, rocke_ptr_type(b, pointee, "global"), &po);
    };
    if(s->softcap)
        (void)rocke_b_param(b, "softcap", rocke_f32(), NULL);
    if(s->sinks)
        in_ptr("sink_ptr", elem, 2);
    if(s->alibi)
        in_ptr("alibi_slopes_ptr", rocke_f32(), 4);
    if(s->qq_bias)
    {
        in_ptr("qq_bias_ptr", rocke_f32(), 4);
        (void)rocke_b_param(b, "qq_bias_rows", rocke_i32(), NULL);
        (void)rocke_b_param(b, "qq_bias_cols", rocke_i32(), NULL);
        (void)rocke_b_param(b, "qq_bias_stride", rocke_i32(), NULL);
    }
    if(eq(s->layout, "ragged"))
    {
        in_ptr("cu_seqlens_q", rocke_i32(), 4);
        in_ptr("cu_seqlens_k", rocke_i32(), 4);
    }
    else if(eq(s->layout, "paged"))
    {
        in_ptr("cu_seqlens_q", rocke_i32(), 4);
        in_ptr("seqused_k", rocke_i32(), 4);
        in_ptr("block_table", rocke_i32(), 4);
        (void)rocke_b_param(b, "block_table_stride", rocke_i32(), NULL);
        (void)rocke_b_param(b, "stride_k_block", rocke_i32(), NULL);
        (void)rocke_b_param(b, "stride_v_block", rocke_i32(), NULL);
        (void)rocke_b_param(b, "num_blocks", rocke_i32(), NULL);
    }
    if(fp8)
    {
        (void)rocke_b_param(b, "k_scale", rocke_f32(), NULL);
        (void)rocke_b_param(b, "v_scale", rocke_f32(), NULL);
    }
    if(dec_has_bias(s))
    {
        const bool f32 = dec_bias_f32(s);
        in_ptr("attn_bias_ptr", f32 ? rocke_f32() : elem, f32 ? 4 : 2);
        (void)rocke_b_param(b, "bias_stride_b", rocke_i32(), NULL);
        (void)rocke_b_param(b, "bias_stride_h", rocke_i32(), NULL);
        (void)rocke_b_param(b, "bias_stride_q", rocke_i32(), NULL);
    }
}

/* Merge kernel parameters (_declare_merge_params). */
void dec_declare_merge_params(rocke_ir_builder_t* b, const rocke_wmma_fmha_decode_spec_t* s)
{
    rocke_param_opts_t o;
    o = dec_opts(true, 2, 8);
    (void)rocke_b_param(b, "O", rocke_ptr_type(b, dec_elem(s), "global"), &o);
    o = dec_opts(true, 2, 4);
    (void)rocke_b_param(b, "LSE", rocke_ptr_type(b, rocke_f32(), "global"), &o);
    o = dec_opts(true, 1, 16);
    (void)rocke_b_param(b, "WS_O", rocke_ptr_type(b, rocke_f32(), "global"), &o);
    o = dec_opts(true, 1, 4);
    (void)rocke_b_param(b, "WS_LSE", rocke_ptr_type(b, rocke_f32(), "global"), &o);
    const char* scalars[]
        = {"seqlen_q", "num_query_heads", "head_dim_v", "num_splits", "write_lse"};
    for(const char* name : scalars)
        (void)rocke_b_param(b, name, rocke_i32(), NULL);
    const char* tensors[] = {"o", "lse"};
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
    if(dec_packed(s))
    {
        o = dec_opts(true, 1, 4);
        (void)rocke_b_param(b, "cu_seqlens_q", rocke_ptr_type(b, rocke_i32(), "global"), &o);
    }
}

/* Small emission helpers shared by the decode and merge bodies. */
struct Em
{
    rocke_ir_builder_t* b;

    Val ci(int64_t v) const
    {
        return rocke_b_const_i32(b, v);
    }
    Val cf(double v) const
    {
        return rocke_b_const_f32(b, v);
    }
    Val add(Val x, Val y) const
    {
        return rocke_b_add(b, x, y);
    }
    Val sub(Val x, Val y) const
    {
        return rocke_b_sub(b, x, y);
    }
    Val mul(Val x, Val y) const
    {
        return rocke_b_mul(b, x, y);
    }
    Val div(Val x, Val y) const
    {
        return rocke_b_div(b, x, y);
    }
    Val mod(Val x, Val y) const
    {
        return rocke_b_mod(b, x, y);
    }
    Val sel(Val c, Val x, Val y) const
    {
        return rocke_b_select(b, c, x, y);
    }
    Val land(Val x, Val y) const
    {
        return rocke_b_land(b, x, y);
    }
    Val smem_ld(Val smem, Val idx, const rocke_type_t* t, int n) const
    {
        return rocke_b_smem_load_vN(b, smem, &idx, 1, t, n);
    }
    void smem_st(Val smem, Val idx, Val v, int n) const
    {
        rocke_b_smem_store_vN(b, smem, &idx, 1, v, n);
    }
    const rocke_type_t* vt(const rocke_type_t* e, int n) const
    {
        return rocke_vector_type(b, e, n);
    }
    Val pack(const Vals& v, const rocke_type_t* e) const
    {
        return rocke_b_vec_pack(b, v.data(), (int)v.size(), e);
    }
    /* tree(vals, op): pairwise reduction in the Python order. */
    template <typename Op>
    Val tree(Vals vals, Op op) const
    {
        while(vals.size() > 1)
        {
            Vals nxt;
            for(size_t i = 0; i + 1 < vals.size(); i += 2)
                nxt.push_back(op(vals[i], vals[i + 1]));
            if(vals.size() % 2)
                nxt.push_back(vals.back());
            vals = nxt;
        }
        return vals[0];
    }
    void if_begin(Val cond) const
    {
        rocke_if_t f = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, f.then_region);
    }
    void if_end() const
    {
        rocke_b_region_leave(b);
    }
};

/* _split_merge: online LSE merge of `ns` split partials of one row. */
void dec_split_merge(const Em& e,
                     Val ws_o,
                     Val ws_lse,
                     Val row_ns,
                     Val ns,
                     Val dv,
                     const Vals& cols,
                     Val* m_out,
                     Val* l_out,
                     Vals* accs_out)
{
    rocke_ir_builder_t* b = e.b;
    Val c0 = e.ci(0);
    Val c1 = e.ci(1);
    Val f0 = e.cf(0.0);
    Val ninf = e.cf(-INFINITY);
    const int U = DEC_MERGE_UNROLL;
    Val n_grp;
    {
        Val t = e.ci(U - 1);
        Val a = e.add(ns, t);
        Val tu = e.ci(U);
        n_grp = e.div(a, tu);
    }
    std::vector<std::string> names = {"mg_m", "mg_l"};
    Vals inits = {ninf, f0};
    for(size_t k = 0; k < cols.size(); ++k)
    {
        names.push_back("mg_a" + std::to_string(k));
        inits.push_back(rocke_b_zero_vec_f32(b, DEC_MERGE_COLS));
    }
    std::vector<rocke_iter_arg_t> iter_args(names.size());
    for(size_t i = 0; i < names.size(); ++i)
    {
        iter_args[i].name = names[i].c_str();
        iter_args[i].init = inits[i];
    }
    rocke_for_t loop = rocke_b_scf_for_iter(
        b, c0, n_grp, c1, iter_args.data(), (int)iter_args.size(), "sg", false, true);
    if(loop.op == NULL)
        return;
    rocke_b_region_enter(b, loop.body);
    {
        Val sg = loop.iv;
        Val* st = loop.iter_vars;
        Vals lses;
        std::vector<Vals> parts;
        for(int i = 0; i < U; ++i)
        {
            Val s;
            {
                Val tu = e.ci(U);
                Val m = e.mul(sg, tu);
                Val ti = e.ci(i);
                s = e.add(m, ti);
            }
            Val ok = rocke_b_cmp_lt(b, s, ns);
            Val prow;
            {
                Val sl = e.sel(ok, s, c0);
                prow = e.add(row_ns, sl);
            }
            {
                Val gl = rocke_b_global_load_f32(b, ws_lse, prow, 4);
                lses.push_back(e.sel(ok, gl, ninf));
            }
            Val base = e.mul(prow, dv);
            Vals pr;
            for(Val cv : cols)
            {
                Val a = e.add(base, cv);
                pr.push_back(rocke_b_global_load_vN(b, ws_o, a, rocke_f32(), DEC_MERGE_COLS, 16));
            }
            parts.push_back(pr);
        }
        Val m_new = st[0];
        for(Val x : lses)
            m_new = rocke_b_fmax(b, m_new, x);
        Val m_sf;
        {
            Val c = rocke_b_fcmp(b, "oeq", m_new, ninf);
            m_sf = e.sel(c, f0, m_new);
        }
        Val alpha;
        {
            Val d = rocke_b_fsub(b, st[0], m_sf);
            alpha = rocke_b_exp2_fast(b, d);
        }
        Val l_new = rocke_b_fmul(b, st[1], alpha);
        Val av = rocke_b_vector_splat(b, alpha, DEC_MERGE_COLS);
        Vals accs;
        for(size_t k = 0; k < cols.size(); ++k)
            accs.push_back(rocke_b_fmul(b, st[2 + k], av));
        for(size_t i = 0; i < lses.size(); ++i)
        {
            Val wgt;
            {
                Val d = rocke_b_fsub(b, lses[i], m_sf);
                wgt = rocke_b_exp2_fast(b, d);
            }
            l_new = rocke_b_fadd(b, l_new, wgt);
            Val wv = rocke_b_vector_splat(b, wgt, DEC_MERGE_COLS);
            for(size_t k = 0; k < accs.size(); ++k)
            {
                Val m = rocke_b_fmul(b, parts[i][k], wv);
                accs[k] = rocke_b_fadd(b, accs[k], m);
            }
        }
        Vals y = {m_new, l_new};
        y.insert(y.end(), accs.begin(), accs.end());
        rocke_b_scf_yield(b, y.data(), (int)y.size());
    }
    rocke_b_region_leave(b);
    Val* res = loop.op->results;
    Val m_all = res[0];
    {
        Val c = rocke_b_fcmp(b, "oeq", m_all, ninf);
        *m_out = e.sel(c, f0, m_all);
    }
    *l_out = res[1];
    accs_out->clear();
    for(size_t k = 0; k < cols.size(); ++k)
        accs_out->push_back(res[2 + k]);
}

/* _fp8x8: eight OCP E4M3FN bytes (<2 x i32>) -> four dwords of `elem` pairs. */
Vals dec_fp8x8(const Em& e, Val raw, const rocke_type_t* elem, bool native)
{
    rocke_ir_builder_t* b = e.b;
    const bool bf16 = elem == rocke_bf16();
    if(native)
    {
        Val v8 = rocke_b_vec_bitcast(b, raw, e.vt(rocke_fp8e4m3(), 8));
        Vals f;
        for(int h = 0; h < 2; ++h)
        {
            Vals ex;
            for(int i = 0; i < 4; ++i)
                ex.push_back(rocke_b_vec_extract(b, v8, 4 * h + i));
            Val q4 = e.pack(ex, rocke_fp8e4m3());
            Val f4 = rocke_b_cvt_pk_f32_fp8x4(b, q4);
            for(int i = 0; i < 4; ++i)
                f.push_back(rocke_b_vec_extract(b, f4, i));
        }
        if(bf16)
        {
            Vals fi;
            for(Val x : f)
                fi.push_back(rocke_b_bitcast(b, x, rocke_i32()));
            Vals out;
            for(int k = 0; k < 4; ++k)
            {
                Val s = e.ci(0x07060302);
                out.push_back(rocke_b_perm_b32(b, fi[2 * k + 1], fi[2 * k], s));
            }
            return out;
        }
        Val pk = e.pack(f, rocke_f32());
        Val c = rocke_b_vec_cast_f32_to(b, pk, rocke_f16());
        Val h16 = rocke_b_vec_bitcast(b, c, e.vt(rocke_i32(), 4));
        Vals out;
        for(int k = 0; k < 4; ++k)
            out.push_back(rocke_b_vec_extract(b, h16, k));
        return out;
    }
    Val m80 = e.ci(0x80808080LL);
    Vals out;
    for(int w = 0; w < 2; ++w)
    {
        Val x = rocke_b_vec_extract(b, raw, w);
        Val mag;
        {
            Val t = e.ci(0x7F7F7F7F);
            mag = e.land(x, t);
        }
        Val hm;
        {
            Val t = e.ci(1);
            hm = rocke_b_lshr(b, mag, t);
        }
        Val hi;
        {
            Val a = e.land(x, m80);
            Val t = e.ci(0x3F3F3F3F);
            Val c = e.land(hm, t);
            hi = rocke_b_lor(b, a, c);
        }
        Val nan;
        {
            Val t = e.ci(0x01010101);
            Val a = e.add(hm, t);
            Val t6 = e.ci(6);
            Val s = rocke_b_shl(b, x, t6);
            Val l = e.land(a, s);
            Val t4 = e.ci(0x40404040);
            nan = e.land(l, t4);
        }
        hi = rocke_b_lor(b, hi, nan);
        Val lo;
        {
            Val t = e.ci(7);
            Val s = rocke_b_shl(b, x, t);
            lo = e.land(s, m80);
        }
        {
            Val s = e.ci(0x05010400);
            out.push_back(rocke_b_perm_b32(b, hi, lo, s));
        }
        {
            Val s = e.ci(0x07030602);
            out.push_back(rocke_b_perm_b32(b, hi, lo, s));
        }
    }
    if(bf16)
    {
        Val pk = e.pack(out, rocke_i32());
        Val h = rocke_b_vec_bitcast(b, pk, e.vt(rocke_f16(), 8));
        Vals f;
        for(int i = 0; i < 8; ++i)
        {
            Val x = rocke_b_vec_extract(b, h, i);
            Val c = rocke_b_cast_to_f32(b, x);
            f.push_back(rocke_b_bitcast(b, c, rocke_i32()));
        }
        Vals o2;
        for(int k = 0; k < 4; ++k)
        {
            Val s = e.ci(0x07060302);
            o2.push_back(rocke_b_perm_b32(b, f[2 * k + 1], f[2 * k], s));
        }
        out = o2;
    }
    return out;
}

/* The Python build_wmma_fmha_decode body after resolution and validation. */
rocke_status_t dec_emit_body(rocke_ir_builder_t* b,
                             const rocke_wmma_fmha_decode_spec_t* s,
                             const char* op_id,
                             bool gfx12)
{
    const Em e{b};
    const rocke_type_t* elem = dec_elem(s);
    const int NW = s->num_waves;
    const int T = DEC_TILE;
    const int NSUB = T / 16;
    const int NP = T / 2;
    const int VG = 32 / NP;
    const int D = s->head_size;
    const int DVS = s->dv_split;
    const int DVT = D / DVS;
    const int n_ks = D / 16;
    const int n_dc = DVT / 16;
    const bool q_lds = eq(s->q_mode, "lds");
    const bool v_early = s->v_store_early == 1;
    const bool pf = s->prefetch == 1;
    const bool lockstep = s->lockstep == 1;
    const bool fp8 = dec_fp8(s);
    const bool packed = dec_packed(s);
    const bool natural = dec_natural(s);
    const int RS = NP + DEC_V_PAD;
    int REG = 0, q_dw = 0;
    dec_lds_dwords(s, &REG, &q_dw);
    const int bucket_min = D == 64 ? 16 : (D == 128 ? 72 : 136);
    const int block_size = 32 * NW;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", block_size);
    if(s->scheduler_strategy != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", s->scheduler_strategy);
    if(s->waves_per_eu != 0)
    {
        const int64_t wpe[2] = {s->waves_per_eu, s->waves_per_eu};
        rocke_attr_set_int_list(b, &b->kernel->attrs, "waves_per_eu", wpe, 2);
    }
    dec_declare_params(b, s);
    auto P = [&](const char* name) { return rocke_b_get_param(b, name); };

    Val c0 = e.ci(0);
    Val c1 = e.ci(1);
    Val c2 = e.ci(2);
    Val c16 = e.ci(16);
    Val f0 = e.cf(0.0);
    Val ninf = e.cf(-INFINITY);
    Val zero8 = rocke_b_zero_vec(b, elem, 8);

    Val Hq = P("num_query_heads");
    Val Hkv = P("num_kv_heads");
    Val Sq = P("seqlen_q");
    Val Sk = P("seqlen_k");
    Val ns = P("num_splits");
    Val d_rt = P("head_dim_q");
    Val dv_rt = P("head_dim_v");

    /* ---- workgroup -> (batch, split, pack group, dv slice, row tile) ---- */
    Val Hv = P("num_v_heads");
    Val n_pg = rocke_b_smax(b, Hkv, Hv);
    Val g = e.div(Hq, n_pg);
    Val rows = e.mul(g, Sq);
    Val n_rt;
    {
        Val t = e.ci(15);
        Val a = e.add(rows, t);
        n_rt = e.div(a, c16);
    }
    Val bid = rocke_b_block_id_x(b);
    Val rt = e.mod(bid, n_rt);
    Val rest = e.div(bid, n_rt);
    Val dvi = c0;
    if(DVS > 1)
    {
        Val t = e.ci(DVS);
        dvi = e.mod(rest, t);
        Val t2 = e.ci(DVS);
        rest = e.div(rest, t2);
    }
    Val pg = e.mod(rest, n_pg);
    rest = e.div(rest, n_pg);
    Val split = e.mod(rest, ns);
    Val batch = e.div(rest, ns);
    Val kvh;
    {
        Val t = e.div(n_pg, Hkv);
        kvh = e.div(pg, t);
    }
    Val vh;
    {
        Val t = e.div(n_pg, Hv);
        vh = e.div(pg, t);
    }
    Val dv0;
    {
        Val t = e.ci(DVT);
        dv0 = e.mul(dvi, t);
    }

    Val tid = rocke_b_thread_id_x(b);
    Val wave;
    {
        Val t = e.ci(32);
        Val d = e.div(tid, t);
        wave = rocke_b_readfirstlane(b, d);
    }
    Val lane;
    {
        Val t = e.ci(32);
        lane = e.mod(tid, t);
    }
    Val col = e.mod(lane, c16);
    Val half = e.div(lane, c16);
    Val lo_half = rocke_b_cmp_lt(b, lane, c16);

    /* ---- per-sequence lengths ---- */
    Val q_start = NULL, k_start = NULL;
    Val Sq_b, Sk_b, rows_b, Sq_d;
    if(packed)
    {
        Val cuq = P("cu_seqlens_q");
        q_start = rocke_b_global_load_i32(b, cuq, batch, 4);
        {
            Val a = e.add(batch, c1);
            Val l = rocke_b_global_load_i32(b, cuq, a, 4);
            Sq_b = e.sub(l, q_start);
        }
        if(eq(s->layout, "ragged"))
        {
            k_start = rocke_b_global_load_i32(b, P("cu_seqlens_k"), batch, 4);
            Val a = e.add(batch, c1);
            Val l = rocke_b_global_load_i32(b, P("cu_seqlens_k"), a, 4);
            Sk_b = e.sub(l, k_start);
        }
        else
            Sk_b = rocke_b_global_load_i32(b, P("seqused_k"), batch, 4);
        rows_b = e.mul(g, Sq_b);
        Sq_d = rocke_b_smax(b, Sq_b, c1);
    }
    else
    {
        Sq_b = Sq;
        Sk_b = Sk;
        rows_b = rows;
        Sq_d = Sq;
    }

    Val r;
    {
        Val m = e.mul(rt, c16);
        r = e.add(m, col);
    }
    Val row_ok = rocke_b_cmp_lt(b, r, rows_b);
    r = e.sel(row_ok, r, c0);
    Val hq;
    {
        Val m = e.mul(pg, g);
        Val d = e.div(r, Sq_d);
        hq = e.add(m, d);
    }
    Val qpos = e.mod(r, Sq_d);

    /* ---- mask ---- */
    Val ctx;
    {
        Val c = rocke_b_cmp_ne(b, P("bottom_right"), c0);
        Val d = e.sub(Sk_b, Sq_b);
        ctx = e.sel(c, d, c0);
    }
    Val wl = P("window_left");
    Val wr = P("window_right");
    Val has_l = rocke_b_cmp_ge(b, wl, c0);
    Val has_r = rocke_b_cmp_ge(b, wr, c0);
    Val diag = e.add(qpos, ctx);
    Val first;
    {
        Val t = e.sub(diag, wl);
        Val m = rocke_b_smax(b, t, c0);
        first = e.sel(has_l, m, c0);
    }
    Val end;
    {
        Val t = e.add(diag, wr);
        Val t2 = e.add(t, c1);
        Val m = rocke_b_smin(b, t2, Sk_b);
        end = e.sel(has_r, m, Sk_b);
    }
    Val span;
    {
        Val t = e.sub(end, first);
        span = rocke_b_smax(b, t, c0);
    }
    span = e.sel(row_ok, span, c0);
    Val span64 = rocke_b_zext(b, span, rocke_i64());
    Val lo;
    {
        Val t = e.sub(ctx, wl);
        Val m = rocke_b_smax(b, t, c0);
        lo = e.sel(has_l, m, c0);
    }
    Val hi;
    {
        Val t = e.add(ctx, wr);
        t = e.add(t, Sq_b);
        t = rocke_b_smin(b, t, Sk_b);
        t = rocke_b_smax(b, t, c0);
        hi = e.sel(has_r, t, Sk_b);
    }
    if(packed)
    {
        Val m = e.mul(rt, c16);
        Val c = rocke_b_cmp_lt(b, m, rows_b);
        hi = e.sel(c, hi, c0);
    }
    Val lo_t;
    {
        Val t = e.ci(T);
        lo_t = e.div(lo, t);
    }
    Val hi_t;
    {
        Val t = e.ci(T - 1);
        Val a = e.add(hi, t);
        Val t2 = e.ci(T);
        hi_t = e.div(a, t2);
    }
    Val step_v;
    {
        Val t = e.ci(NW);
        step_v = e.mul(ns, t);
    }
    Val first_t;
    {
        Val t = e.ci(NW);
        Val m = e.mul(split, t);
        first_t = e.add(lo_t, m);
    }
    Val my_t0 = e.add(first_t, wave);
    Val my_end = hi_t;
    Val wg_n;
    {
        Val t = e.sub(hi_t, first_t);
        Val m = rocke_b_smax(b, t, c0);
        Val s1 = e.sub(step_v, c1);
        Val a = e.add(m, s1);
        wg_n = e.div(a, step_v);
    }
    Val n_it;
    if(lockstep)
        n_it = wg_n;
    else
    {
        Val t = e.sub(my_end, my_t0);
        Val m = rocke_b_smax(b, t, c0);
        Val s1 = e.sub(step_v, c1);
        Val a = e.add(m, s1);
        n_it = e.div(a, step_v);
    }

    /* ---- pointers ---- */
    const int kv_eb = fp8 ? 1 : 2;
    auto kv_bytes = [&](Val x) { return kv_eb == 1 ? x : e.mul(x, c2); };
    auto scaled_ptr = [&](Val ptr, Val idx, Val stride, int elem_bytes) {
        Val z1 = rocke_b_zext(b, idx, rocke_i64());
        Val z2 = rocke_b_zext(b, stride, rocke_i64());
        Val m = e.mul(z1, z2);
        Val c = rocke_b_const_i64(b, elem_bytes);
        Val off = e.mul(m, c);
        return rocke_b_global_ptr_add(b, ptr, off);
    };
    Val Qb, Ob;
    if(packed)
    {
        /* An empty trailing sequence starts one past the last Q row; its (dead) Q
         * loads read the first row instead. */
        Val nonempty = rocke_b_cmp_gt(b, Sq_b, c0);
        Val q_base = e.sel(nonempty, q_start, c0);
        Qb = scaled_ptr(P("Q"), q_base, P("stride_q_token"), 2);
        Ob = scaled_ptr(P("O"), q_start, P("stride_o_token"), 2);
    }
    else
    {
        Qb = scaled_ptr(P("Q"), batch, P("stride_q_batch"), 2);
        Ob = scaled_ptr(P("O"), batch, P("stride_o_batch"), 2);
    }
    Val k_ss = P("stride_k_token");
    Val v_ss = P("stride_v_token");
    auto kv_base = [&](const char* ptr, const char* sb, Val ss) -> Val {
        if(eq(s->layout, "dense"))
            return scaled_ptr(P(ptr), batch, P(sb), kv_eb);
        if(eq(s->layout, "ragged"))
            return scaled_ptr(P(ptr), k_start, ss, kv_eb);
        return P(ptr);
    };
    Val k_slice;
    {
        Val base = kv_base("K", "stride_k_batch", k_ss);
        Val m = e.mul(kvh, P("stride_k_head"));
        Val kb = kv_bytes(m);
        k_slice = rocke_b_global_ptr_add(b, base, kb);
    }
    Val v_slice;
    {
        Val base = kv_base("V", "stride_v_batch", v_ss);
        Val m = e.mul(vh, P("stride_v_head"));
        Val kb = kv_bytes(m);
        v_slice = rocke_b_global_ptr_add(b, base, kb);
    }
    Val sk1 = e.sub(Sk_b, c1);
    Val k_extent;
    {
        Val m = e.mul(sk1, k_ss);
        Val a = e.add(m, d_rt);
        k_extent = kv_bytes(a);
    }
    Val v_extent;
    {
        Val m = e.mul(sk1, v_ss);
        Val t = e.ci(DVT);
        Val a2 = e.add(dv0, t);
        Val a = e.add(m, a2);
        v_extent = kv_bytes(a);
    }
    {
        Val m = e.mul(sk1, v_ss);
        Val a = e.add(m, dv_rt);
        Val kb = kv_bytes(a);
        v_extent = rocke_b_smin(b, v_extent, kb);
    }
    Val k_row_bytes = kv_bytes(k_ss);
    Val v_row_bytes = kv_bytes(v_ss);
    Val k_tile_bytes;
    {
        Val t = e.ci(T);
        k_tile_bytes = e.mul(k_row_bytes, t);
    }
    Val v_tile_bytes;
    {
        Val t = e.ci(T);
        v_tile_bytes = e.mul(v_row_bytes, t);
    }

    auto tile_rsrc = [&](Val slice_ptr, Val extent, Val tile_bytes, Val t, Val valid) {
        Val off = e.mul(t, tile_bytes);
        Val d = e.sub(extent, off);
        Val nrec = rocke_b_smax(b, d, c0);
        if(valid != NULL)
            nrec = e.sel(valid, nrec, c0);
        Val p = rocke_b_global_ptr_add(b, slice_ptr, off);
        return rocke_b_buffer_rsrc(b, p, nrec);
    };

    const bool paged = eq(s->layout, "paged");
    const bool page_rows = paged && s->page_block_size < T;
    Val page_shift = NULL, page_mask = NULL, bt_row = NULL;
    if(paged)
    {
        const int ps = s->page_block_size;
        int shift = 0;
        while((1 << (shift + 1)) <= ps)
            ++shift;
        page_shift = e.ci(shift);
        page_mask = e.ci(ps - 1);
        bt_row = e.mul(batch, P("block_table_stride"));
    }
    auto page_of = [&](Val key, Val valid) {
        Val blk = rocke_b_lshr(b, key, page_shift);
        if(valid != NULL)
            blk = e.sel(valid, blk, c0);
        Val a = e.add(bt_row, blk);
        return rocke_b_global_load_i32(b, P("block_table"), a, 4);
    };
    /* `phys` names a page of the cache: 0 <= phys < num_blocks. The table's
     * values are caller data; a page outside the cache reads as zeros. */
    auto page_ok = [&](Val phys) {
        Val ge = rocke_b_cmp_ge(b, phys, c0);
        Val lt = rocke_b_cmp_lt(b, phys, P("num_blocks"));
        return rocke_b_land(b, ge, lt);
    };
    auto paged_tile_rsrc
        = [&](Val slice_ptr, Val extent, Val row_bytes, Val blk_stride, Val t, Val valid) {
              Val kv0_;
              {
                  Val tt = e.ci(T);
                  kv0_ = e.mul(t, tt);
              }
              Val phys;
              {
                  Val po = page_of(kv0_, valid);
                  phys = rocke_b_readfirstlane(b, po);
              }
              Val in_cache = page_ok(phys);
              Val page_off;
              {
                  Val z1 = rocke_b_zext(b, phys, rocke_i64());
                  Val z2 = rocke_b_zext(b, blk_stride, rocke_i64());
                  Val m = e.mul(z1, z2);
                  Val c = rocke_b_const_i64(b, kv_eb);
                  page_off = e.mul(m, c);
              }
              {
                  Val la = e.land(kv0_, page_mask);
                  Val m = e.mul(la, row_bytes);
                  Val z = rocke_b_zext(b, m, rocke_i64());
                  page_off = e.add(page_off, z);
              }
              Val nrec;
              {
                  Val m = e.mul(kv0_, row_bytes);
                  Val d = e.sub(extent, m);
                  nrec = rocke_b_smax(b, d, c0);
              }
              if(valid != NULL)
                  nrec = e.sel(valid, nrec, c0);
              nrec = e.sel(in_cache, nrec, c0);
              Val p = rocke_b_global_ptr_add(b, slice_ptr, page_off);
              return rocke_b_buffer_rsrc(b, p, nrec);
          };
    auto k_rsrc = [&](Val t, Val valid) {
        if(paged)
            return paged_tile_rsrc(k_slice, k_extent, k_row_bytes, P("stride_k_block"), t, valid);
        return tile_rsrc(k_slice, k_extent, k_tile_bytes, t, valid);
    };
    auto v_rsrc = [&](Val t, Val valid) {
        if(paged)
            return paged_tile_rsrc(v_slice, v_extent, v_row_bytes, P("stride_v_block"), t, valid);
        return tile_rsrc(v_slice, v_extent, v_tile_bytes, t, valid);
    };

    Vals k_lane;
    for(int a = 0; a < NSUB; ++a)
    {
        Val t = e.ci(a * 16);
        Val ad = e.add(t, col);
        k_lane.push_back(e.mul(ad, k_row_bytes));
    }
    for(auto& x : k_lane)
    {
        Val m;
        if(kv_eb == 2)
            m = e.mul(half, c16);
        else
        {
            Val t = e.ci(8);
            m = e.mul(half, t);
        }
        x = e.add(x, m);
    }
    Val vp;
    {
        Val t = e.ci(NP);
        vp = e.mod(lane, t);
    }
    Val vcg;
    {
        Val t = e.ci(NP);
        vcg = e.div(lane, t);
    }
    Val v_lane0;
    {
        Val m1 = e.mul(vp, c2);
        Val m2 = e.mul(m1, v_row_bytes);
        Val t = e.ci(8);
        Val m3 = e.mul(vcg, t);
        Val a = e.add(m3, dv0);
        Val kb = kv_bytes(a);
        v_lane0 = e.add(m2, kb);
    }
    Vals v_lane = {v_lane0, e.add(v_lane0, v_row_bytes)};
    const int n_vj = DVT / (8 * VG);

    Val k_all = NULL, v_all = NULL, k_col_lane = NULL, v_col_lane = NULL;
    if(page_rows)
    {
        Val all_recs = e.ci(0x7FFF0000);
        k_all = rocke_b_buffer_rsrc(b, k_slice, all_recs);
        v_all = rocke_b_buffer_rsrc(b, v_slice, all_recs);
        {
            Val t = e.ci(8);
            Val m = e.mul(half, t);
            k_col_lane = kv_bytes(m);
        }
        {
            Val t = e.ci(8);
            Val m = e.mul(vcg, t);
            Val a = e.add(m, dv0);
            v_col_lane = kv_bytes(a);
        }
    }
    auto row_offs
        = [&](Val t, Val valid, const Vals& rel_rows, Val blk_stride, Val row_stride, Val col_off) {
              Val kv0_;
              {
                  Val tt = e.ci(T);
                  kv0_ = e.mul(t, tt);
              }
              Vals out;
              for(Val rel : rel_rows)
              {
                  Val key = e.add(kv0_, rel);
                  Val ok = rocke_b_cmp_lt(b, key, Sk_b);
                  if(valid != NULL)
                      ok = e.land(ok, valid);
                  Val phys = page_of(key, ok);
                  {
                      Val in_cache = page_ok(phys);
                      ok = e.land(ok, in_cache);
                  }
                  Val off;
                  {
                      Val m1 = e.mul(phys, blk_stride);
                      Val la = e.land(key, page_mask);
                      Val m2 = e.mul(la, row_stride);
                      off = e.add(m1, m2);
                  }
                  Val kb = kv_bytes(off);
                  Val a = e.add(kb, col_off);
                  Val dead = e.ci(0x7FFF8000);
                  out.push_back(e.sel(ok, a, dead));
              }
              return out;
          };
    auto k_row_offs = [&](Val t, Val valid) {
        Vals rel;
        for(int a = 0; a < NSUB; ++a)
        {
            Val ta = e.ci(a * 16);
            rel.push_back(e.add(ta, col));
        }
        return row_offs(t, valid, rel, P("stride_k_block"), k_ss, k_col_lane);
    };
    auto v_row_offs = [&](Val t, Val valid) {
        Vals rel;
        rel.push_back(e.mul(vp, c2));
        {
            Val m = e.mul(vp, c2);
            rel.push_back(e.add(m, c1));
        }
        return row_offs(t, valid, rel, P("stride_v_block"), v_ss, v_col_lane);
    };

    /* ---- Q: B operand of S^T = K Q^T ---- */
    Val q_off;
    {
        Val m1 = e.mul(hq, P("stride_q_head"));
        Val m2 = e.mul(qpos, P("stride_q_token"));
        q_off = e.add(m1, m2);
    }
    auto q_part = [&](int cbase, bool per_lane) {
        Val colv = e.ci(cbase);
        if(per_lane)
        {
            Val t = e.ci(8);
            Val m = e.mul(half, t);
            colv = e.add(colv, m);
        }
        Val ok;
        {
            Val c = rocke_b_cmp_lt(b, colv, d_rt);
            ok = e.land(row_ok, c);
        }
        Val raw;
        {
            Val a = e.add(q_off, colv);
            Val sl = e.sel(ok, a, c0);
            raw = rocke_b_global_load_vN(b, Qb, sl, elem, 8, 16);
        }
        return e.sel(ok, raw, zero8);
    };
    Val q_smem = NULL;
    Val q_rd = NULL;
    const int QRS = D / 2 + 4;
    auto q_pack = [&](int ks) -> Val {
        if(!q_lds)
        {
            if(gfx12)
                return q_part(ks * 16, true);
            Val x = q_part(ks * 16, false);
            Val y = q_part(ks * 16 + 8, false);
            return rocke_b_vec_concat(b, x, y);
        }
        Val base;
        {
            Val t = e.ci(ks * 8);
            base = e.add(q_rd, t);
        }
        if(gfx12)
        {
            Val ld = e.smem_ld(q_smem, base, rocke_i32(), 4);
            return rocke_b_vec_bitcast(b, ld, e.vt(elem, 8));
        }
        Val lo_ = e.smem_ld(q_smem, base, rocke_i32(), 4);
        Val a4;
        {
            Val t = e.ci(4);
            a4 = e.add(base, t);
        }
        Val hi_ = e.smem_ld(q_smem, a4, rocke_i32(), 4);
        Val cat = rocke_b_vec_concat(b, lo_, hi_);
        return rocke_b_vec_bitcast(b, cat, e.vt(elem, 16));
    };
    Vals q_packs;
    if(!q_lds)
    {
        for(int ks = 0; ks < n_ks; ++ks)
            q_packs.push_back(q_pack(ks));
    }
    else
    {
        {
            const int shape[1] = {DEC_ROWS * QRS};
            q_smem = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "dec_q");
        }
        const int per_row = D / 8;
        const int n_ch = DEC_ROWS * per_row;
        for(int k = 0; k < (n_ch + block_size - 1) / block_size; ++k)
        {
            Val ch;
            {
                Val t = e.ci(k * block_size);
                ch = e.add(tid, t);
            }
            Val ch_ok;
            {
                Val t = e.ci(n_ch);
                ch_ok = rocke_b_cmp_lt(b, ch, t);
            }
            Val qrow;
            {
                Val t = e.ci(per_row);
                qrow = e.div(ch, t);
            }
            Val c8;
            {
                Val t = e.ci(per_row);
                c8 = e.mod(ch, t);
            }
            Val rq;
            {
                Val m = e.mul(rt, c16);
                rq = e.add(m, qrow);
            }
            Val rq_ok = rocke_b_cmp_lt(b, rq, rows_b);
            rq = e.sel(rq_ok, rq, c0);
            Val colq;
            {
                Val t = e.ci(8);
                colq = e.mul(c8, t);
            }
            Val ok;
            {
                Val l1 = e.land(ch_ok, rq_ok);
                Val c = rocke_b_cmp_lt(b, colq, d_rt);
                ok = e.land(l1, c);
            }
            Val off;
            {
                Val m1 = e.mul(pg, g);
                Val d1 = e.div(rq, Sq_d);
                Val a1 = e.add(m1, d1);
                Val m2 = e.mul(a1, P("stride_q_head"));
                Val md = e.mod(rq, Sq_d);
                Val m3 = e.mul(md, P("stride_q_token"));
                Val a2 = e.add(m2, m3);
                off = e.add(a2, colq);
            }
            Val raw;
            {
                Val sl = e.sel(ok, off, c0);
                Val ld = rocke_b_global_load_vN(b, Qb, sl, elem, 8, 16);
                raw = e.sel(ok, ld, zero8);
            }
            e.if_begin(ch_ok);
            {
                Val t = e.ci(QRS);
                Val m1 = e.mul(qrow, t);
                Val t4 = e.ci(4);
                Val m2 = e.mul(c8, t4);
                Val idx = e.add(m1, m2);
                Val v = rocke_b_vec_bitcast(b, raw, e.vt(rocke_i32(), 4));
                e.smem_st(q_smem, idx, v, 4);
            }
            e.if_end();
        }
        rocke_b_sync(b);
        {
            Val t = e.ci(QRS);
            q_rd = e.mul(col, t);
        }
        if(gfx12)
        {
            Val t = e.ci(4);
            Val m = e.mul(half, t);
            q_rd = e.add(q_rd, m);
        }
    }

    /* ---- bias ---- */
    const bool has_bias = dec_has_bias(s);
    const rocke_type_t* bias_elem = dec_bias_f32(s) ? rocke_f32() : elem;
    const int bias_bytes = dec_bias_f32(s) ? 4 : 2;
    Val bias_rsrc = NULL, bias_row = NULL;
    const bool bias_q = has_bias && !dec_bias_f32(s);
    Val bias_par = NULL, bias_sel = NULL;
    if(has_bias)
    {
        Val bias_base;
        {
            Val z1 = rocke_b_zext(b, batch, rocke_i64());
            Val z2 = rocke_b_zext(b, P("bias_stride_b"), rocke_i64());
            Val m = e.mul(z1, z2);
            Val c = rocke_b_const_i64(b, bias_bytes);
            Val m2 = e.mul(m, c);
            bias_base = rocke_b_global_ptr_add(b, P("attn_bias_ptr"), m2);
        }
        Val ext;
        {
            Val s1 = e.sub(Hq, c1);
            Val m1 = e.mul(s1, P("bias_stride_h"));
            Val s2 = e.sub(Sq, c1);
            Val m2 = e.mul(s2, P("bias_stride_q"));
            Val a = e.add(m1, m2);
            ext = e.add(a, Sk);
        }
        {
            Val t = e.ci(bias_bytes);
            Val m = e.mul(ext, t);
            Val lead_bytes = NULL;
            if(bias_q)
            {
                /* A Q-dtype resource may start or end mid-dword, and the
                 * hardware drops a loaded dword that ends past num_records:
                 * start at the dword below the slice and span whole dwords
                 * (rows sit lead_bytes / 2 elements in). */
                Val addr = rocke_b_global_ptr_to_i64(b, bias_base);
                Val lo = rocke_b_trunc(b, addr, rocke_i32());
                Val t2 = e.ci(2);
                lead_bytes = e.land(lo, t2);
                Val z = rocke_b_const_i64(b, 0);
                Val lead64 = rocke_b_zext(b, lead_bytes, rocke_i64());
                Val back = e.sub(z, lead64);
                bias_base = rocke_b_global_ptr_add(b, bias_base, back);
                Val s1 = e.add(m, lead_bytes);
                Val t3 = e.ci(3);
                Val s2 = e.add(s1, t3);
                Val tm = e.ci(-4);
                m = e.land(s2, tm);
            }
            bias_rsrc = rocke_b_buffer_rsrc(b, bias_base, m);
            {
                Val m1 = e.mul(hq, P("bias_stride_h"));
                Val m2 = e.mul(qpos, P("bias_stride_q"));
                bias_row = e.add(m1, m2);
            }
            if(bias_q)
            {
                /* Each row is read from its dword-aligned key; bias_par is the
                 * 16-bit half this lane's first key sits in (tile and sub-tile
                 * offsets are even). */
                Val lead = rocke_b_lshr(b, lead_bytes, c1);
                Val row = e.add(bias_row, lead);
                if(!gfx12)
                    row = e.add(row, half);
                bias_par = e.land(row, c1);
                bias_row = e.sub(row, bias_par);
                if(gfx12)
                {
                    Val odd = rocke_b_cmp_ne(b, bias_par, c0);
                    Val shifted = e.ci(0x05040302);
                    Val plain = e.ci(0x03020100);
                    bias_sel = e.sel(odd, shifted, plain);
                }
            }
        }
    }
    auto bias_vals = [&](Val kv0, int a) {
        Val first_key;
        {
            Val t = e.ci(a * 16);
            first_key = e.add(kv0, t);
        }
        const int n = gfx12 ? 8 : 16;
        Val base = e.add(bias_row, first_key);
        if(gfx12)
        {
            Val t = e.ci(8);
            Val m = e.mul(half, t);
            base = e.add(base, m);
        }
        if(bias_q && gfx12)
        {
            /* Five dwords from the aligned key, shifted by bias_par halves. */
            Val t = e.ci(bias_bytes);
            Val off = e.mul(base, t);
            Val raw = rocke_b_buffer_load_vN(b, bias_rsrc, off, c0, rocke_i32(), 4);
            Vals words;
            for(int i = 0; i < 4; ++i)
                words.push_back(rocke_b_vec_extract(b, raw, i));
            Val t16 = e.ci(16);
            Val off1 = e.add(off, t16);
            raw = rocke_b_buffer_load_vN(b, bias_rsrc, off1, c0, rocke_i32(), 1);
            words.push_back(rocke_b_vec_extract(b, raw, 0));
            Vals shifted;
            for(int i = 0; i < 4; ++i)
                shifted.push_back(rocke_b_perm_b32(b, words[i + 1], words[i], bias_sel));
            Val pk = e.pack(shifted, rocke_i32());
            Val v8 = rocke_b_vec_bitcast(b, pk, e.vt(elem, 8));
            Vals out;
            for(int i = 0; i < 8; ++i)
            {
                Val x = rocke_b_vec_extract(b, v8, i);
                out.push_back(rocke_b_cast_to_f32(b, x));
            }
            return out;
        }
        const int per = dec_bias_f32(s) ? 4 : 8;
        Vals vals;
        for(int c = 0; c < n / per; ++c)
        {
            Val voff;
            {
                Val t = e.ci(c * per);
                Val a1 = e.add(base, t);
                Val t2 = e.ci(bias_bytes);
                voff = e.mul(a1, t2);
            }
            Val raw = rocke_b_buffer_load_vN(b, bias_rsrc, voff, c0, bias_elem, per);
            for(int i = 0; i < per; ++i)
                vals.push_back(rocke_b_vec_extract(b, raw, i));
        }
        if(bias_elem != rocke_f32())
        {
            for(auto& v : vals)
                v = rocke_b_cast_to_f32(b, v);
        }
        if(gfx12)
            return vals;
        Val odd = rocke_b_cmp_ne(b, bias_q ? bias_par : half, c0);
        Vals out;
        for(int k = 0; k < 8; ++k)
            out.push_back(e.sel(odd, vals[2 * k + 1], vals[2 * k]));
        return out;
    };

    /* ---- score units ---- */
    Val qk_scale = P("scale_log2");
    Val v_mul = NULL;
    if(fp8)
    {
        Val unit = e.cf(s->fp8_cvt == 1 ? 1.0 : DEC_FP8_INT_UNIT);
        Val m = rocke_b_fmul(b, P("k_scale"), unit);
        qk_scale = rocke_b_fmul(b, qk_scale, m);
        v_mul = rocke_b_fmul(b, P("v_scale"), unit);
    }
    Val scale_nat = NULL, cs;
    if(natural)
    {
        Val t = e.cf(1.0 / DEC_LOG2E);
        scale_nat = rocke_b_fmul(b, qk_scale, t);
        cs = e.cf(DEC_LOG2E);
    }
    else
        cs = qk_scale;
    auto slot_key = [&](int a, int el) { return a * 16 + (gfx12 ? el : 2 * el); };
    Val lane_key;
    if(gfx12)
    {
        Val t = e.ci(8);
        lane_key = e.mul(half, t);
    }
    else
        lane_key = half;

    Val cap = NULL, cap_in = NULL, cap_e = NULL, cap_m2 = NULL, f1 = NULL;
    const bool tanh_fast = s->tanh_fast == 1;
    if(s->softcap)
    {
        cap = P("softcap");
        cap_in = rocke_b_fdiv(b, scale_nat, cap);
        if(tanh_fast)
        {
            Val t = e.cf(2.0 * DEC_LOG2E);
            cap_e = rocke_b_fmul(b, cap_in, t);
            Val t2 = e.cf(-2.0);
            cap_m2 = rocke_b_fmul(b, cap, t2);
            f1 = e.cf(1.0);
        }
    }
    Val slope = NULL;
    if(s->alibi)
        slope = rocke_b_global_load(b, P("alibi_slopes_ptr"), hq, rocke_f32(), 4);
    Val qq_rsrc = NULL, qq_q_ok = NULL, qq_row = NULL, qq_cols64 = NULL;
    if(s->qq_bias)
    {
        Val qq_cols = P("qq_bias_cols");
        Val qq_stride = P("qq_bias_stride");
        Val qq_ext;
        {
            Val s1 = e.sub(P("qq_bias_rows"), c1);
            Val m = e.mul(s1, qq_stride);
            qq_ext = e.add(m, qq_cols);
        }
        {
            Val mx = rocke_b_smax(b, qq_ext, c0);
            Val t = e.ci(4);
            Val m = e.mul(mx, t);
            qq_rsrc = rocke_b_buffer_rsrc(b, P("qq_bias_ptr"), m);
        }
        {
            Val c = rocke_b_cmp_lt(b, qpos, P("qq_bias_rows"));
            qq_q_ok = e.land(row_ok, c);
        }
        qq_row = e.mul(qpos, qq_stride);
        qq_cols64 = rocke_b_zext(b, qq_cols, rocke_i64());
    }

    /* ---- LDS ---- */
    Val smem;
    {
        const int shape[1] = {NW * REG};
        smem = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "dec_lds");
    }
    Val wbase;
    {
        Val t = e.ci(REG);
        wbase = e.mul(wave, t);
    }
    Val vt_wr;
    {
        Val t = e.ci(8);
        Val m1 = e.mul(vcg, t);
        Val t2 = e.ci(RS);
        Val m2 = e.mul(m1, t2);
        Val a = e.add(m2, vp);
        vt_wr = e.add(wbase, a);
    }
    Val vt_rd;
    {
        Val t = e.ci(RS);
        Val m = e.mul(col, t);
        vt_rd = e.add(wbase, m);
    }
    if(gfx12)
    {
        Val t = e.ci(4);
        Val m = e.mul(half, t);
        vt_rd = e.add(vt_rd, m);
    }
    Val sel_lo;
    {
        Val t1 = e.ci(0x05040100);
        Val t2 = e.ci(0x01000504);
        sel_lo = e.sel(lo_half, t1, t2);
    }
    Val sel_hi;
    {
        Val t1 = e.ci(0x07060302);
        Val t2 = e.ci(0x03020706);
        sel_hi = e.sel(lo_half, t1, t2);
    }
    auto peer_f32 = [&](Val v) {
        Val bc = rocke_b_bitcast(b, v, rocke_i32());
        Val pl = rocke_b_permlanex16(b, bc);
        return rocke_b_bitcast(b, pl, rocke_f32());
    };
    auto fmax_op = [&](Val x, Val y) { return rocke_b_fmax(b, x, y); };
    auto fadd_op = [&](Val x, Val y) { return rocke_b_fadd(b, x, y); };
    Val big_off = e.ci(0x7FFFFFF0);
    auto live_off = [&](Val voff, Val colv, Val width) {
        Val c = rocke_b_cmp_lt(b, colv, width);
        return e.sel(c, voff, big_off);
    };

    /* feature_adds(kv0): per-tile additive terms of slot (a, e). */
    Val rel0_f = NULL, al0 = NULL;
    auto feature_begin = [&](Val kv0) {
        if(!(s->alibi || s->qq_bias))
            return;
        Val a = e.add(kv0, lane_key);
        rel0_f = e.sub(a, ctx);
        if(s->alibi)
        {
            Val f = rocke_b_sitofp_f32(b, rel0_f);
            al0 = rocke_b_fmul(b, slope, f);
        }
    };
    auto feature_add = [&](int a, int el) -> Val {
        if(!(s->alibi || s->qq_bias))
            return NULL;
        const int k = slot_key(a, el);
        Val out = NULL;
        if(s->alibi)
        {
            if(k != 0)
            {
                Val c = e.cf((double)k);
                out = rocke_b_fma(b, slope, c, al0);
            }
            else
                out = al0;
        }
        if(s->qq_bias)
        {
            Val relk;
            {
                Val t = e.ci(k);
                relk = e.add(rel0_f, t);
            }
            Val ok;
            {
                Val z = rocke_b_zext(b, relk, rocke_i64());
                Val c = rocke_b_cmp_lt(b, z, qq_cols64);
                ok = e.land(qq_q_ok, c);
            }
            Val off;
            {
                Val a1 = e.add(qq_row, relk);
                Val t4 = e.ci(4);
                Val m = e.mul(a1, t4);
                off = e.sel(ok, m, big_off);
            }
            Val ld = rocke_b_buffer_load_vN(b, qq_rsrc, off, c0, rocke_f32(), 1);
            Val qv = rocke_b_vec_extract(b, ld, 0);
            out = out == NULL ? qv : rocke_b_fadd(b, out, qv);
        }
        return out;
    };

    auto k_parts = [&](Val rsrc, const Vals& lanes) {
        Vals out;
        for(int a = 0; a < NSUB; ++a)
        {
            for(int ks = 0; ks < n_ks; ++ks)
            {
                const int cbase = ks * 16;
                Val voff;
                {
                    Val t = e.ci(cbase * kv_eb);
                    voff = e.add(lanes[a], t);
                }
                if(cbase + 16 > bucket_min)
                {
                    Val t = e.ci(cbase);
                    Val t8 = e.ci(8);
                    Val m = e.mul(half, t8);
                    Val colv = e.add(t, m);
                    voff = live_off(voff, colv, d_rt);
                }
                if(fp8)
                    out.push_back(rocke_b_buffer_load_vN(b, rsrc, voff, c0, rocke_i32(), 2));
                else
                    out.push_back(rocke_b_buffer_load_vN(b, rsrc, voff, c0, elem, 8));
            }
        }
        return out;
    };
    auto k_load = [&](Val t, Val valid) {
        if(page_rows)
        {
            Vals offs = k_row_offs(t, valid);
            return k_parts(k_all, offs);
        }
        Val r_ = k_rsrc(t, valid);
        return k_parts(r_, k_lane);
    };
    const bool fp8_cvt = s->fp8_cvt == 1;
    auto k_dec = [&](Val raw) {
        if(!fp8)
            return raw;
        Vals d = dec_fp8x8(e, raw, elem, fp8_cvt);
        Val pk = e.pack(d, rocke_i32());
        return rocke_b_vec_bitcast(b, pk, e.vt(elem, 8));
    };
    auto k_operand = [&](const Vals& parts, int a, int ks) {
        const int i = a * n_ks + ks;
        if(gfx12)
            return k_dec(parts[i]);
        Val own;
        {
            Val kd = k_dec(parts[i]);
            own = rocke_b_vec_bitcast(b, kd, e.vt(rocke_i32(), 4));
        }
        Vals o_, pr, lo_, hi_;
        for(int j = 0; j < 4; ++j)
            o_.push_back(rocke_b_vec_extract(b, own, j));
        for(int j = 0; j < 4; ++j)
            pr.push_back(rocke_b_permlanex16(b, o_[j]));
        for(int j = 0; j < 4; ++j)
            lo_.push_back(e.sel(lo_half, o_[j], pr[j]));
        for(int j = 0; j < 4; ++j)
            hi_.push_back(e.sel(lo_half, pr[j], o_[j]));
        lo_.insert(lo_.end(), hi_.begin(), hi_.end());
        Val pk = e.pack(lo_, rocke_i32());
        return rocke_b_vec_bitcast(b, pk, e.vt(elem, 16));
    };
    auto v_rows = [&](Val rsrc, const Vals& lanes, int j, Vals* out) {
        for(int rr = 0; rr < 2; ++rr)
        {
            Val voff;
            {
                Val t = e.ci(j * VG * 8 * kv_eb);
                voff = e.add(lanes[rr], t);
            }
            if(DVS > 1 || 8 * VG * (j + 1) > bucket_min)
            {
                Val t1 = e.ci(8 * VG * j);
                Val t8 = e.ci(8);
                Val m = e.mul(vcg, t8);
                Val a = e.add(t1, m);
                Val colv = e.add(dv0, a);
                voff = live_off(voff, colv, dv_rt);
            }
            if(fp8)
                out->push_back(rocke_b_buffer_load_vN(b, rsrc, voff, c0, rocke_i32(), 2));
            else
            {
                Val ld = rocke_b_buffer_load_vN(b, rsrc, voff, c0, elem, 8);
                out->push_back(rocke_b_vec_bitcast(b, ld, e.vt(rocke_i32(), 4)));
            }
        }
    };
    auto v_store = [&](const Vals& vflat) {
        for(int j = 0; j < n_vj; ++j)
        {
            Val ra = vflat[2 * j], rb = vflat[2 * j + 1];
            Vals fa, fb;
            if(fp8)
            {
                fa = dec_fp8x8(e, ra, elem, fp8_cvt);
                fb = dec_fp8x8(e, rb, elem, fp8_cvt);
            }
            for(int k = 0; k < 4; ++k)
            {
                Val xa, xb;
                if(fp8)
                {
                    xa = fa[k];
                    xb = fb[k];
                }
                else
                {
                    xa = rocke_b_vec_extract(b, ra, k);
                    xb = rocke_b_vec_extract(b, rb, k);
                }
                const int sels[2] = {0x05040100, 0x07060302};
                for(int tt = 0; tt < 2; ++tt)
                {
                    const int d_rel = 8 * VG * j + 2 * k + tt;
                    Val t = e.ci(d_rel * RS);
                    Val idx = e.add(vt_wr, t);
                    Val sl = e.ci(sels[tt]);
                    Val pm = rocke_b_perm_b32(b, xb, xa, sl);
                    e.smem_st(smem, idx, pm, 1);
                }
            }
        }
    };
    auto p_operand = [&](const Vals& ps8) {
        if(gfx12)
        {
            Val pk = e.pack(ps8, rocke_f32());
            return rocke_b_vec_cast_f32_to(b, pk, elem);
        }
        Vals outs;
        for(int m = 0; m < 4; ++m)
        {
            Val lo_, hi_;
            {
                Val c = rocke_b_cast_f32_to(b, ps8[2 * m], elem);
                Val bc = rocke_b_bitcast(b, c, rocke_i16());
                lo_ = rocke_b_zext(b, bc, rocke_i32());
            }
            {
                Val c = rocke_b_cast_f32_to(b, ps8[2 * m + 1], elem);
                Val bc = rocke_b_bitcast(b, c, rocke_i16());
                hi_ = rocke_b_zext(b, bc, rocke_i32());
            }
            Val v;
            {
                Val sh = rocke_b_shl(b, hi_, c16);
                v = rocke_b_lor(b, lo_, sh);
            }
            Val w = rocke_b_permlanex16(b, v);
            outs.push_back(rocke_b_perm_b32(b, w, v, sel_lo));
            outs.push_back(rocke_b_perm_b32(b, w, v, sel_hi));
        }
        Val pk = e.pack(outs, rocke_i32());
        return rocke_b_vec_bitcast(b, pk, e.vt(elem, 16));
    };
    auto v_load = [&](Val t, Val valid) {
        Val rsrc;
        Vals lanes;
        if(page_rows)
        {
            rsrc = v_all;
            lanes = v_row_offs(t, valid);
        }
        else
        {
            rsrc = v_rsrc(t, valid);
            lanes = v_lane;
        }
        Vals flat;
        for(int j = 0; j < n_vj; ++j)
            v_rows(rsrc, lanes, j, &flat);
        return flat;
    };

    /* ---- key loop ---- */
    std::vector<std::string> names = {"m", "l"};
    Vals inits = {ninf, f0};
    for(int d = 0; d < n_dc; ++d)
    {
        names.push_back("o" + std::to_string(d));
        inits.push_back(rocke_b_zero_vec_f32(b, 8));
    }
    size_t n_kp = 0, n_vp = 0;
    if(pf)
    {
        Val ok0 = rocke_b_cmp_lt(b, my_t0, my_end);
        Vals kp0 = k_load(my_t0, ok0);
        Vals vp0 = v_load(my_t0, ok0);
        n_kp = kp0.size();
        n_vp = vp0.size();
        for(size_t i = 0; i < kp0.size(); ++i)
        {
            names.push_back("pk" + std::to_string(i));
            inits.push_back(kp0[i]);
        }
        for(size_t i = 0; i < vp0.size(); ++i)
        {
            names.push_back("pv" + std::to_string(i));
            inits.push_back(vp0[i]);
        }
    }
    std::vector<rocke_iter_arg_t> iter_args(names.size());
    for(size_t i = 0; i < names.size(); ++i)
    {
        iter_args[i].name = names[i].c_str();
        iter_args[i].init = inits[i];
    }
    rocke_for_t loop = rocke_b_scf_for_iter(
        b, c0, n_it, c1, iter_args.data(), (int)iter_args.size(), "it", false, true);
    if(loop.op == NULL)
        return rocke_ir_builder_status(b);
    rocke_b_region_enter(b, loop.body);
    {
        Val it = loop.iv;
        Val* st = loop.iter_vars;
        Val m_i = st[0], l_i = st[1];
        Vals accs(st + 2, st + 2 + n_dc);
        Val t;
        {
            Val m = e.mul(it, step_v);
            t = e.add(my_t0, m);
        }
        Val kv0;
        {
            Val tt = e.ci(T);
            kv0 = e.mul(t, tt);
        }
        Vals kp, vflat;
        Val t_n = NULL, ok_n = NULL;
        if(pf)
        {
            kp.assign(st + 2 + n_dc, st + 2 + n_dc + n_kp);
            vflat.assign(st + 2 + n_dc + n_kp, st + 2 + n_dc + n_kp + n_vp);
            t_n = e.add(t, step_v);
            ok_n = rocke_b_cmp_lt(b, t_n, my_end);
        }
        else
        {
            Val act = lockstep ? rocke_b_cmp_lt(b, t, my_end) : NULL;
            kp = k_load(t, act);
            vflat = v_load(t, act);
        }
        Vals s_acc;
        for(int a = 0; a < NSUB; ++a)
            s_acc.push_back(rocke_b_zero_vec_f32(b, 8));
        for(int ks = 0; ks < n_ks; ++ks)
        {
            Val q_k = q_lds ? q_pack(ks) : q_packs[ks];
            for(int a = 0; a < NSUB; ++a)
            {
                Val ko = k_operand(kp, a, ks);
                s_acc[a] = rocke_b_mma(b, op_id, ko, q_k, s_acc[a], NULL, 0);
            }
        }
        Vals kp_next, vp_next;
        if(pf)
            kp_next = k_load(t_n, ok_n);
        if(v_early)
        {
            v_store(vflat);
            if(pf)
                vp_next = v_load(t_n, ok_n);
        }

        /* scores: features, mask, online softmax */
        Val rel0;
        {
            Val d = e.sub(kv0, first);
            rel0 = e.add(d, lane_key);
        }
        Val span_t = span64;
        if(lockstep)
        {
            Val c = rocke_b_cmp_lt(b, t, my_end);
            Val z = rocke_b_const_i64(b, 0);
            span_t = e.sel(c, span64, z);
        }
        feature_begin(kv0);
        Vals svals;
        Vals bvs;
        for(int a = 0; a < NSUB; ++a)
        {
            for(int el = 0; el < 8; ++el)
            {
                Val sc = rocke_b_vec_extract(b, s_acc[a], el);
                if(natural)
                {
                    Val extra = feature_add(a, el);
                    if(has_bias)
                    {
                        if(el == 0)
                            bvs = bias_vals(kv0, a);
                        extra = extra == NULL ? bvs[el] : rocke_b_fadd(b, extra, bvs[el]);
                    }
                    if(s->softcap)
                    {
                        if(tanh_fast)
                        {
                            Val m = rocke_b_fmul(b, sc, cap_e);
                            Val e2x = rocke_b_exp2_fast(b, m);
                            Val a1 = rocke_b_fadd(b, e2x, f1);
                            Val rc = rocke_b_rcp_fast(b, a1);
                            sc = rocke_b_fma(b, cap_m2, rc, cap);
                        }
                        else
                        {
                            Val m = rocke_b_fmul(b, sc, cap_in);
                            Val th = rocke_b_tanh(b, m);
                            sc = rocke_b_fmul(b, cap, th);
                        }
                        if(extra != NULL)
                            sc = rocke_b_fadd(b, sc, extra);
                    }
                    else
                        sc = rocke_b_fma(b, sc, scale_nat, extra);
                }
                Val rel;
                {
                    Val tk = e.ci(slot_key(a, el));
                    Val ad = e.add(rel0, tk);
                    rel = rocke_b_zext(b, ad, rocke_i64());
                }
                Val c = rocke_b_cmp_lt(b, rel, span_t);
                svals.push_back(e.sel(c, sc, ninf));
            }
        }
        Val m_loc = e.tree(svals, fmax_op);
        Val m_new;
        {
            Val pr = peer_f32(m_loc);
            Val mx = rocke_b_fmax(b, m_loc, pr);
            m_new = rocke_b_fmax(b, m_i, mx);
        }
        Val m_safe;
        {
            Val c = rocke_b_fcmp(b, "oeq", m_new, ninf);
            m_safe = e.sel(c, f0, m_new);
        }
        Val nmc;
        {
            Val m = rocke_b_fmul(b, cs, m_safe);
            nmc = rocke_b_fneg(b, m);
        }
        Val alpha;
        {
            Val f = rocke_b_fma(b, m_i, cs, nmc);
            alpha = rocke_b_exp2_fast(b, f);
        }
        Vals ps;
        for(Val sv : svals)
        {
            Val f = rocke_b_fma(b, sv, cs, nmc);
            ps.push_back(rocke_b_exp2_fast(b, f));
        }
        Val l_loc = e.tree(ps, fadd_op);
        Val l_new;
        {
            Val pr = peer_f32(l_loc);
            Val a = rocke_b_fadd(b, l_loc, pr);
            l_new = rocke_b_fma(b, l_i, alpha, a);
        }
        Val alpha_v = rocke_b_vector_splat(b, alpha, 8);
        for(auto& acc : accs)
            acc = rocke_b_fmul(b, acc, alpha_v);
        Vals p_ops;
        for(int a = 0; a < NSUB; ++a)
        {
            Vals ps8(ps.begin() + a * 8, ps.begin() + a * 8 + 8);
            p_ops.push_back(p_operand(ps8));
        }
        if(!v_early)
        {
            v_store(vflat);
            if(pf)
                vp_next = v_load(t_n, ok_n);
        }

        /* O^T += V^T P */
        for(int dc = 0; dc < n_dc; ++dc)
        {
            for(int a = 0; a < NSUB; ++a)
            {
                Val base;
                {
                    Val tt = e.ci(dc * 16 * RS + a * 8);
                    base = e.add(vt_rd, tt);
                }
                Val vt;
                if(gfx12)
                {
                    Val ld = e.smem_ld(smem, base, rocke_i32(), 4);
                    vt = rocke_b_vec_bitcast(b, ld, e.vt(elem, 8));
                }
                else
                {
                    Val lo_ = e.smem_ld(smem, base, rocke_i32(), 4);
                    Val a4;
                    {
                        Val t4 = e.ci(4);
                        a4 = e.add(base, t4);
                    }
                    Val hi_ = e.smem_ld(smem, a4, rocke_i32(), 4);
                    Val cat = rocke_b_vec_concat(b, lo_, hi_);
                    vt = rocke_b_vec_bitcast(b, cat, e.vt(elem, 16));
                }
                accs[dc] = rocke_b_mma(b, op_id, vt, p_ops[a], accs[dc], NULL, 0);
            }
        }
        if(lockstep)
            rocke_b_sync(b);
        Vals y = {m_new, l_new};
        y.insert(y.end(), accs.begin(), accs.end());
        if(pf)
        {
            y.insert(y.end(), kp_next.begin(), kp_next.end());
            y.insert(y.end(), vp_next.begin(), vp_next.end());
        }
        rocke_b_scf_yield(b, y.data(), (int)y.size());
    }
    rocke_b_region_leave(b);
    Val m_w = loop.op->results[0];
    Val l_w = loop.op->results[1];
    Vals o_w(loop.op->results + 2, loop.op->results + 2 + n_dc);

    /* ---- combine the waves through LDS ---- */
    {
        Val rec_ml;
        {
            Val m = e.mul(lane, c2);
            rec_ml = e.add(wbase, m);
        }
        Val bm = rocke_b_bitcast(b, m_w, rocke_i32());
        Val bl = rocke_b_bitcast(b, l_w, rocke_i32());
        Val pk = e.pack(Vals{bm, bl}, rocke_i32());
        e.smem_st(smem, rec_ml, pk, 2);
    }
    for(int dc = 0; dc < n_dc; ++dc)
    {
        Val frag = rocke_b_vec_bitcast(b, o_w[dc], e.vt(rocke_i32(), 8));
        Val idx;
        {
            Val t = e.ci(64 + dc * 256);
            Val t8 = e.ci(8);
            Val m = e.mul(lane, t8);
            Val a = e.add(t, m);
            idx = e.add(wbase, a);
        }
        {
            Vals ex;
            for(int i = 0; i < 4; ++i)
                ex.push_back(rocke_b_vec_extract(b, frag, i));
            Val pk = e.pack(ex, rocke_i32());
            e.smem_st(smem, idx, pk, 4);
        }
        {
            Val t4 = e.ci(4);
            Val a = e.add(idx, t4);
            Vals ex;
            for(int i = 4; i < 8; ++i)
                ex.push_back(rocke_b_vec_extract(b, frag, i));
            Val pk = e.pack(ex, rocke_i32());
            e.smem_st(smem, a, pk, 4);
        }
    }
    rocke_b_sync(b);

    Vals ms, ls;
    for(int w2 = 0; w2 < NW; ++w2)
    {
        Val t = e.ci(w2 * REG);
        Val m = e.mul(lane, c2);
        Val a = e.add(t, m);
        Val ml = e.smem_ld(smem, a, rocke_i32(), 2);
        {
            Val x = rocke_b_vec_extract(b, ml, 0);
            ms.push_back(rocke_b_bitcast(b, x, rocke_f32()));
        }
        {
            Val x = rocke_b_vec_extract(b, ml, 1);
            ls.push_back(rocke_b_bitcast(b, x, rocke_f32()));
        }
    }
    Val m_all = e.tree(ms, fmax_op);
    Val m_all_safe;
    {
        Val c = rocke_b_fcmp(b, "oeq", m_all, ninf);
        m_all_safe = e.sel(c, f0, m_all);
    }
    Val nmc;
    {
        Val m = rocke_b_fmul(b, cs, m_all_safe);
        nmc = rocke_b_fneg(b, m);
    }
    Vals fs;
    for(Val mw : ms)
    {
        Val f = rocke_b_fma(b, mw, cs, nmc);
        fs.push_back(rocke_b_exp2_fast(b, f));
    }
    Val l_all;
    {
        Vals prods;
        for(int w2 = 0; w2 < NW; ++w2)
            prods.push_back(rocke_b_fmul(b, ls[w2], fs[w2]));
        l_all = e.tree(prods, fadd_op);
    }
    Val has, inv_l, lse2;
    if(s->sinks)
    {
        Val sink2;
        {
            Val ld = rocke_b_global_load(b, P("sink_ptr"), hq, elem, 2);
            Val f = rocke_b_cast_to_f32(b, ld);
            Val t = e.cf(DEC_LOG2E);
            sink2 = rocke_b_fmul(b, f, t);
        }
        {
            Val c = rocke_b_cmp_eq(b, split, c0);
            sink2 = e.sel(c, sink2, ninf);
        }
        Val m2;
        {
            Val c = rocke_b_fcmp(b, "ogt", l_all, f0);
            Val mm = rocke_b_fmul(b, cs, m_all_safe);
            m2 = e.sel(c, mm, ninf);
        }
        Val top = rocke_b_fmax(b, m2, sink2);
        {
            Val c = rocke_b_fcmp(b, "oeq", top, ninf);
            top = e.sel(c, f0, top);
        }
        Val keep;
        {
            Val d = rocke_b_fsub(b, m2, top);
            keep = rocke_b_exp2_fast(b, d);
        }
        Val den;
        {
            Val m = rocke_b_fmul(b, l_all, keep);
            Val d = rocke_b_fsub(b, sink2, top);
            Val ex = rocke_b_exp2_fast(b, d);
            den = rocke_b_fadd(b, m, ex);
        }
        has = rocke_b_fcmp(b, "ogt", den, f0);
        {
            Val d = rocke_b_fdiv(b, keep, den);
            inv_l = e.sel(has, d, f0);
        }
        {
            Val l = rocke_b_log2(b, den);
            Val a = rocke_b_fadd(b, top, l);
            lse2 = e.sel(has, a, ninf);
        }
    }
    else
    {
        has = rocke_b_fcmp(b, "ogt", l_all, f0);
        {
            Val one = e.cf(1.0);
            Val d = rocke_b_fdiv(b, one, l_all);
            inv_l = e.sel(has, d, f0);
        }
        {
            Val m = rocke_b_fmul(b, cs, m_all_safe);
            Val l = rocke_b_log2(b, l_all);
            Val a = rocke_b_fadd(b, m, l);
            lse2 = e.sel(has, a, ninf);
        }
    }
    if(v_mul != NULL)
        inv_l = rocke_b_fmul(b, inv_l, v_mul);
    Vals fsv;
    for(Val fw : fs)
        fsv.push_back(rocke_b_vector_splat(b, fw, 8));
    Val inv_v = rocke_b_vector_splat(b, inv_l, 8);

    Val single = rocke_b_cmp_eq(b, ns, c1);
    Val prow;
    {
        Val m1 = e.mul(batch, Hq);
        Val a1 = e.add(m1, hq);
        Val m2 = e.mul(a1, Sq);
        Val a2 = e.add(m2, qpos);
        Val m3 = e.mul(a2, ns);
        prow = e.add(m3, split);
    }
    Val o_row;
    {
        Val m1 = e.mul(hq, P("stride_o_head"));
        Val m2 = e.mul(qpos, P("stride_o_token"));
        o_row = e.add(m1, m2);
    }
    Val ws_row = e.mul(prow, dv_rt);

    for(int kk = 0; kk < (n_dc + NW - 1) / NW; ++kk)
    {
        Val dc;
        {
            Val t = e.ci(kk * NW);
            dc = e.add(t, wave);
        }
        Val acc = NULL;
        for(int w2 = 0; w2 < NW; ++w2)
        {
            Val idx;
            {
                Val t1 = e.ci(w2 * REG + 64);
                Val t2 = e.ci(256);
                Val m = e.mul(dc, t2);
                Val a = e.add(t1, m);
                Val t8 = e.ci(8);
                Val ml = e.mul(lane, t8);
                idx = e.add(a, ml);
            }
            Val lo_ = e.smem_ld(smem, idx, rocke_i32(), 4);
            Val a4;
            {
                Val t4 = e.ci(4);
                a4 = e.add(idx, t4);
            }
            Val hi_ = e.smem_ld(smem, a4, rocke_i32(), 4);
            Val cat = rocke_b_vec_concat(b, lo_, hi_);
            Val fr = rocke_b_vec_bitcast(b, cat, e.vt(rocke_f32(), 8));
            Val term = rocke_b_fmul(b, fr, fsv[w2]);
            acc = acc == NULL ? term : rocke_b_fadd(b, acc, term);
        }
        acc = rocke_b_fmul(b, acc, inv_v);
        Vals o;
        for(int i = 0; i < 8; ++i)
            o.push_back(rocke_b_vec_extract(b, acc, i));
        if(!gfx12)
        {
            Vals send, recv, out;
            for(int k = 0; k < 4; ++k)
                send.push_back(e.sel(lo_half, o[4 + k], o[k]));
            for(Val x : send)
                recv.push_back(peer_f32(x));
            for(int k = 0; k < 4; ++k)
            {
                out.push_back(e.sel(lo_half, o[k], recv[k]));
                out.push_back(e.sel(lo_half, recv[k], o[4 + k]));
            }
            o = out;
        }
        Val colv;
        {
            Val m = e.mul(dc, c16);
            Val a = e.add(dv0, m);
            Val t8 = e.ci(8);
            Val m2 = e.mul(half, t8);
            colv = e.add(a, m2);
        }
        Val ok;
        {
            Val c1_ = rocke_b_cmp_lt(b, colv, dv_rt);
            Val l1 = e.land(row_ok, c1_);
            Val t = e.ci(n_dc);
            Val c2_ = rocke_b_cmp_lt(b, dc, t);
            ok = e.land(l1, c2_);
        }
        {
            Val cond = e.land(ok, single);
            e.if_begin(cond);
            Val pk = e.pack(o, rocke_f32());
            Val vec = rocke_b_vec_cast_f32_to(b, pk, elem);
            Val a = e.add(o_row, colv);
            Val m = e.mul(a, c2);
            Val ptr = rocke_b_global_ptr_add(b, Ob, m);
            rocke_b_global_store(b, ptr, c0, vec, 16);
            e.if_end();
        }
        {
            Val n = rocke_b_lnot(b, single);
            Val cond = e.land(ok, n);
            e.if_begin(cond);
            Val a = e.add(ws_row, colv);
            Val a64 = rocke_b_zext(b, a, rocke_i64());
            Val t4 = rocke_b_const_i64(b, 4);
            Val m = e.mul(a64, t4);
            Val ptr = rocke_b_global_ptr_add(b, P("WS_O"), m);
            Val p1 = e.pack(Vals(o.begin(), o.begin() + 4), rocke_f32());
            rocke_b_global_store(b, ptr, c0, p1, 16);
            Val p2 = e.pack(Vals(o.begin() + 4, o.end()), rocke_f32());
            rocke_b_global_store(b, ptr, c1, p2, 16);
            e.if_end();
        }
    }

    auto lse_seq = [&]() {
        if(packed)
            return e.mul(q_start, P("stride_lse_token"));
        return e.mul(batch, P("stride_lse_batch"));
    };
    Val stats;
    {
        Val l1 = e.land(row_ok, lo_half);
        Val c = rocke_b_cmp_eq(b, wave, c0);
        stats = e.land(l1, c);
    }
    {
        Val c = rocke_b_cmp_eq(b, dvi, c0);
        Val l1 = e.land(stats, c);
        Val c2_ = rocke_b_cmp_ne(b, P("write_lse"), c0);
        Val l2 = e.land(single, c2_);
        Val cond = e.land(l1, l2);
        e.if_begin(cond);
        Val ls_ = lse_seq();
        Val m1 = e.mul(hq, P("stride_lse_head"));
        Val m2 = e.mul(qpos, P("stride_lse_token"));
        Val a = e.add(m1, m2);
        Val lse_off = e.add(ls_, a);
        Val t = e.cf(DEC_LN2);
        Val v = rocke_b_fmul(b, lse2, t);
        rocke_b_global_store(b, P("LSE"), lse_off, v, 4);
        e.if_end();
    }
    {
        Val n = rocke_b_lnot(b, single);
        Val cond = e.land(stats, n);
        e.if_begin(cond);
        rocke_b_global_store(b, P("WS_LSE"), prow, lse2, 4);
        e.if_end();
    }

    /* ---- fused merge (WS_CNT non-null) ---- */
    {
        Val i64 = rocke_b_global_ptr_to_i64(b, P("WS_CNT"));
        Val z = rocke_b_const_i64(b, 0);
        Val c = rocke_b_cmp_ne(b, i64, z);
        e.if_begin(c);
        {
            Val n = rocke_b_lnot(b, single);
            e.if_begin(n);
            rocke_b_fence(b, "agent", "release");
            rocke_b_sync(b);
            Val grp;
            {
                Val m1 = e.mul(batch, n_pg);
                Val a1 = e.add(m1, pg);
                Val t = e.ci(DVS);
                Val m2 = e.mul(a1, t);
                Val a2 = e.add(m2, dvi);
                Val m3 = e.mul(a2, n_rt);
                grp = e.add(m3, rt);
            }
            {
                Val c0t = rocke_b_cmp_eq(b, tid, c0);
                e.if_begin(c0t);
                Val old = rocke_b_global_atomic_add(b, P("WS_CNT"), grp, c1, NULL);
                e.smem_st(smem, c0, old, 1);
                e.if_end();
            }
            rocke_b_sync(b);
            Val arrived;
            {
                Val ld = e.smem_ld(smem, c0, rocke_i32(), 1);
                Val x = rocke_b_vec_extract(b, ld, 0);
                arrived = rocke_b_readfirstlane(b, x);
            }
            {
                Val s1 = e.sub(ns, c1);
                Val cl = rocke_b_cmp_eq(b, arrived, s1);
                e.if_begin(cl);
                rocke_b_fence(b, "agent", "acquire");
                const int tpr = block_size / DEC_ROWS;
                Val mrow;
                {
                    Val t = e.ci(tpr);
                    mrow = e.div(tid, t);
                }
                Val mcg;
                {
                    Val t = e.ci(tpr);
                    mcg = e.mod(tid, t);
                }
                Val rm;
                {
                    Val m = e.mul(rt, c16);
                    rm = e.add(m, mrow);
                }
                Val rm_ok = rocke_b_cmp_lt(b, rm, rows_b);
                rm = e.sel(rm_ok, rm, c0);
                Val hq_m;
                {
                    Val m = e.mul(pg, g);
                    Val d = e.div(rm, Sq_d);
                    hq_m = e.add(m, d);
                }
                Val qpos_m = e.mod(rm, Sq_d);
                Val row_ns;
                {
                    Val m1 = e.mul(batch, Hq);
                    Val a1 = e.add(m1, hq_m);
                    Val m2 = e.mul(a1, Sq);
                    Val a2 = e.add(m2, qpos_m);
                    row_ns = e.mul(a2, ns);
                }
                int n_c4 = (DVT / DEC_MERGE_COLS + tpr - 1) / tpr;
                if(n_c4 < 1)
                    n_c4 = 1;
                Vals cols, oks;
                for(int k = 0; k < n_c4; ++k)
                {
                    Val rel;
                    {
                        Val t = e.ci(k * tpr);
                        Val a = e.add(mcg, t);
                        Val t4 = e.ci(DEC_MERGE_COLS);
                        rel = e.mul(a, t4);
                    }
                    Val colv = e.add(dv0, rel);
                    Val ok;
                    {
                        Val t = e.ci(DVT);
                        Val c1_ = rocke_b_cmp_lt(b, rel, t);
                        Val c2_ = rocke_b_cmp_lt(b, colv, dv_rt);
                        Val l = e.land(c1_, c2_);
                        ok = e.land(rm_ok, l);
                    }
                    cols.push_back(e.sel(ok, colv, c0));
                    oks.push_back(ok);
                }
                Val m_sf, l_m;
                Vals accs_m;
                dec_split_merge(
                    e, P("WS_O"), P("WS_LSE"), row_ns, ns, dv_rt, cols, &m_sf, &l_m, &accs_m);
                if(!rocke_ir_builder_ok(b))
                    return rocke_ir_builder_status(b);
                Val has_m = rocke_b_fcmp(b, "ogt", l_m, f0);
                Val inv_lm;
                {
                    Val one = e.cf(1.0);
                    Val d = rocke_b_fdiv(b, one, l_m);
                    Val sl = e.sel(has_m, d, f0);
                    inv_lm = rocke_b_vector_splat(b, sl, DEC_MERGE_COLS);
                }
                Val o_row_m;
                {
                    Val m1 = e.mul(hq_m, P("stride_o_head"));
                    Val m2 = e.mul(qpos_m, P("stride_o_token"));
                    o_row_m = e.add(m1, m2);
                }
                for(size_t k = 0; k < cols.size(); ++k)
                {
                    e.if_begin(oks[k]);
                    Val m = rocke_b_fmul(b, accs_m[k], inv_lm);
                    Val vec = rocke_b_vec_cast_f32_to(b, m, elem);
                    Val a = e.add(o_row_m, cols[k]);
                    Val mm = e.mul(a, c2);
                    Val ptr = rocke_b_global_ptr_add(b, Ob, mm);
                    rocke_b_global_store(b, ptr, c0, vec, 8);
                    e.if_end();
                }
                Val lse_ok;
                {
                    Val c = rocke_b_cmp_eq(b, mcg, c0);
                    Val l1 = e.land(rm_ok, c);
                    Val c2_ = rocke_b_cmp_eq(b, dvi, c0);
                    Val c3 = rocke_b_cmp_ne(b, P("write_lse"), c0);
                    Val l2 = e.land(c2_, c3);
                    lse_ok = e.land(l1, l2);
                }
                {
                    e.if_begin(lse_ok);
                    Val ls_ = lse_seq();
                    Val m1 = e.mul(hq_m, P("stride_lse_head"));
                    Val m2 = e.mul(qpos_m, P("stride_lse_token"));
                    Val a = e.add(m1, m2);
                    Val lse_off = e.add(ls_, a);
                    Val l = rocke_b_log2(b, l_m);
                    Val ad = rocke_b_fadd(b, m_sf, l);
                    Val t = e.cf(DEC_LN2);
                    Val m = rocke_b_fmul(b, ad, t);
                    Val val = e.sel(has_m, m, ninf);
                    rocke_b_global_store(b, P("LSE"), lse_off, val, 4);
                    e.if_end();
                }
                {
                    Val c0t = rocke_b_cmp_eq(b, tid, c0);
                    e.if_begin(c0t);
                    rocke_b_global_store(b, P("WS_CNT"), grp, c0, 4);
                    e.if_end();
                }
                e.if_end();
            }
            e.if_end();
        }
        e.if_end();
    }
    rocke_b_ret(b);
    return rocke_ir_builder_status(b);
}

/* The Python build_wmma_fmha_decode_merge body after validation. */
rocke_status_t dec_emit_merge(rocke_ir_builder_t* b, const rocke_wmma_fmha_decode_spec_t* s)
{
    const Em e{b};
    const bool packed = dec_packed(s);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", DEC_MERGE_THREADS);
    dec_declare_merge_params(b, s);
    auto P = [&](const char* name) { return rocke_b_get_param(b, name); };

    Val c0 = e.ci(0);
    Val c1 = e.ci(1);
    Val f0 = e.cf(0.0);
    Val ninf = e.cf(-INFINITY);
    Val Sq = P("seqlen_q");
    Val Hq = P("num_query_heads");
    Val ns = P("num_splits");
    Val dv = P("head_dim_v");
    Val row = rocke_b_block_id_x(b);
    Val qpos = e.mod(row, Sq);
    Val hq;
    {
        Val d = e.div(row, Sq);
        hq = e.mod(d, Hq);
    }
    Val batch;
    {
        Val m = e.mul(Sq, Hq);
        batch = e.div(row, m);
    }
    Val tid = rocke_b_thread_id_x(b);
    Val d;
    {
        Val t = e.ci(DEC_MERGE_COLS);
        d = e.mul(tid, t);
    }
    Val d_ok = rocke_b_cmp_lt(b, d, dv);
    Val d_safe = e.sel(d_ok, d, c0);
    Val row_ns = e.mul(row, ns);
    Val row_ok = NULL, o_seq = NULL, lse_seq = NULL;
    if(packed)
    {
        Val q_start = rocke_b_global_load_i32(b, P("cu_seqlens_q"), batch, 4);
        {
            Val a = e.add(batch, c1);
            Val l = rocke_b_global_load_i32(b, P("cu_seqlens_q"), a, 4);
            Val sl = e.sub(l, q_start);
            row_ok = rocke_b_cmp_lt(b, qpos, sl);
        }
        o_seq = e.mul(q_start, P("stride_o_token"));
        lse_seq = e.mul(q_start, P("stride_lse_token"));
    }
    Val m_safe, l_all;
    Vals accs;
    dec_split_merge(
        e, P("WS_O"), P("WS_LSE"), row_ns, ns, dv, Vals{d_safe}, &m_safe, &l_all, &accs);
    if(!rocke_ir_builder_ok(b))
        return rocke_ir_builder_status(b);
    Val has = rocke_b_fcmp(b, "ogt", l_all, f0);
    Val inv_l;
    {
        Val one = e.cf(1.0);
        Val dd = rocke_b_fdiv(b, one, l_all);
        inv_l = e.sel(has, dd, f0);
    }
    Val out;
    {
        Val sp = rocke_b_vector_splat(b, inv_l, DEC_MERGE_COLS);
        Val m = rocke_b_fmul(b, accs[0], sp);
        out = rocke_b_vec_cast_f32_to(b, m, dec_elem(s));
    }
    Val o_off;
    {
        Val m1 = o_seq != NULL ? o_seq : e.mul(batch, P("stride_o_batch"));
        Val m2 = e.mul(qpos, P("stride_o_token"));
        Val a1 = e.add(m1, m2);
        Val m3 = e.mul(hq, P("stride_o_head"));
        Val a2 = e.add(m3, d_safe);
        o_off = e.add(a1, a2);
    }
    {
        Val cond = row_ok == NULL ? d_ok : e.land(d_ok, row_ok);
        e.if_begin(cond);
        Val t = e.ci(2);
        Val m = e.mul(o_off, t);
        Val ptr = rocke_b_global_ptr_add(b, P("O"), m);
        rocke_b_global_store(b, ptr, c0, out, 8);
        e.if_end();
    }
    Val lse_ok;
    {
        Val c = rocke_b_cmp_eq(b, tid, c0);
        Val c2 = rocke_b_cmp_ne(b, P("write_lse"), c0);
        lse_ok = e.land(c, c2);
    }
    {
        Val cond = row_ok == NULL ? lse_ok : e.land(lse_ok, row_ok);
        e.if_begin(cond);
        Val lse;
        {
            Val l = rocke_b_log2(b, l_all);
            Val a = rocke_b_fadd(b, m_safe, l);
            Val t = e.cf(DEC_LN2);
            Val m = rocke_b_fmul(b, a, t);
            lse = e.sel(has, m, ninf);
        }
        Val lse_off;
        {
            Val first = lse_seq != NULL ? lse_seq : e.mul(batch, P("stride_lse_batch"));
            Val m1 = e.mul(hq, P("stride_lse_head"));
            Val m2 = e.mul(qpos, P("stride_lse_token"));
            Val a = e.add(m1, m2);
            lse_off = e.add(first, a);
        }
        rocke_b_global_store(b, P("LSE"), lse_off, lse, 4);
        e.if_end();
    }
    rocke_b_ret(b);
    return rocke_ir_builder_status(b);
}

rocke_status_t dec_signature(const rocke_wmma_fmha_decode_spec_t* spec,
                             bool merge,
                             rocke_arena_t* arena,
                             const rocke_sig_entry_t** out_items,
                             size_t* out_count)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL
       || !dec_post_init_ok(spec, reason, sizeof(reason)))
        return ROCKE_ERR_VALUE;
    rocke_ir_builder_t b;
    rocke_status_t status = rocke_ir_builder_init(&b, "rocke_wmma_fmha_decode_sig_probe");
    if(status != ROCKE_OK)
        return status;
    if(merge)
        dec_declare_merge_params(&b, spec);
    else
        dec_declare_params(&b, spec);
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

bool fits_i32(int64_t n)
{
    return 0 < n && n <= 0x7FFFFFFF;
}

} // namespace

rocke_wmma_fmha_decode_spec_t rocke_wmma_fmha_decode_spec_default(void)
{
    rocke_wmma_fmha_decode_spec_t s;
    s.head_size = 128;
    s.dtype = "fp16";
    s.bias = "";
    s.softcap = false;
    s.sinks = false;
    s.alibi = false;
    s.qq_bias = false;
    s.kv_dtype = "";
    s.layout = "dense";
    s.page_block_size = 0;
    s.num_waves = 4;
    s.dv_split = 0;
    s.q_mode = "";
    s.v_store_early = -1;
    s.prefetch = -1;
    s.lockstep = -1;
    s.scheduler_strategy = "auto";
    s.waves_per_eu = 0;
    s.fp8_cvt = -1;
    s.tanh_fast = -1;
    s.wmma_family = "";
    s.name = DEC_DEFAULT_NAME;
    return s;
}

int rocke_wmma_fmha_decode_block_size(const rocke_wmma_fmha_decode_spec_t* spec)
{
    return 32 * spec->num_waves;
}

rocke_status_t rocke_wmma_fmha_decode_resolve(const rocke_wmma_fmha_decode_spec_t* spec,
                                              const char* arch,
                                              rocke_wmma_fmha_decode_spec_t* out,
                                              char* reason,
                                              size_t reason_cap)
{
    if(spec == NULL || out == NULL || arch == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec/arch/out");
        return ROCKE_ERR_VALUE;
    }
    rocke_wmma_fmha_decode_spec_t r;
    if(!dec_resolve(spec, arch, &r, NULL, NULL, reason, reason_cap))
        return ROCKE_ERR_VALUE;
    *out = r;
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_fmha_decode_kernel_name(const rocke_wmma_fmha_decode_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(spec == NULL || out == NULL || !dec_post_init_ok(spec, reason, sizeof(reason)))
        return ROCKE_ERR_VALUE;
    const rocke_wmma_fmha_decode_spec_t* s = spec;
    std::string n = s->name != NULL ? s->name : DEC_DEFAULT_NAME;
    n += "_d" + std::to_string(s->head_size);
    n += std::string("_") + s->dtype;
    n += "_w" + std::to_string(s->num_waves);
    if(dec_has_bias(s))
        n += std::string("_bias") + s->bias;
    if(s->softcap)
        n += "_cap";
    if(s->sinks)
        n += "_sink";
    if(s->alibi)
        n += "_alibi";
    if(s->qq_bias)
        n += "_qq";
    if(dec_fp8(s))
        n += "_kvfp8";
    if(eq(s->layout, "ragged"))
        n += "_ragged";
    else if(eq(s->layout, "paged"))
        n += "_paged" + std::to_string(s->page_block_size);
    if(s->dv_split != 0)
        n += "_dvs" + std::to_string(s->dv_split);
    const char* fam = sz(s->wmma_family);
    if(fam[0] != '\0')
        n += "_t" + std::to_string(DEC_TILE);
    if(sz(s->q_mode)[0] != '\0')
        n += std::string("_q") + s->q_mode;
    if(eq(fam, "gfx11"))
        n += "_khalf";
    if(s->v_store_early >= 0)
        n += s->v_store_early ? "_vearly" : "_vlate";
    if(s->prefetch >= 0)
        n += s->prefetch ? "_pf" : "_nopf";
    if(fam[0] != '\0')
        n += "_si";
    if(s->lockstep == 1)
        n += "_ls";
    if(!eq(s->scheduler_strategy, "auto"))
    {
        std::string sched = s->scheduler_strategy != NULL ? s->scheduler_strategy : "default";
        std::string tag;
        for(char ch : sched)
            if(ch != '-')
                tag += ch;
        n += "_" + tag;
    }
    if(s->waves_per_eu != 0)
        n += "_wpe" + std::to_string(s->waves_per_eu);
    if(dec_fp8(s) && s->fp8_cvt >= 0)
        n += s->fp8_cvt ? "_cvt8" : "_int8";
    if(s->softcap && s->tanh_fast >= 0)
        n += s->tanh_fast ? "_tanhf" : "_tanhm";
    if(n.size() + 1 > out_cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_fmha_decode_merge_kernel_name(const rocke_wmma_fmha_decode_spec_t* spec,
                                                        char* out,
                                                        size_t out_cap)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(spec == NULL || out == NULL || !dec_post_init_ok(spec, reason, sizeof(reason)))
        return ROCKE_ERR_VALUE;
    std::string n = std::string("rocke_wmma_fmha_decode_merge_") + spec->dtype;
    if(dec_packed(spec))
        n += "_packed";
    if(n.size() + 1 > out_cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

bool rocke_wmma_fmha_decode_is_valid_spec(const rocke_wmma_fmha_decode_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap)
{
    rocke_wmma_fmha_decode_spec_t r;
    std::string op;
    bool gfx12 = false;
    return dec_valid(spec, arch, &r, &op, &gfx12, reason, reason_cap);
}

int rocke_wmma_fmha_decode_lds_bytes(const rocke_wmma_fmha_decode_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    rocke_wmma_fmha_decode_spec_t r;
    if(spec == NULL || arch == NULL
       || !dec_resolve(spec, arch, &r, NULL, NULL, reason, sizeof(reason)))
        return -1;
    return dec_lds_bytes_resolved(&r);
}

rocke_kernel_def_t* rocke_build_wmma_fmha_decode(rocke_ir_builder_t* b,
                                                 const rocke_wmma_fmha_decode_spec_t* spec,
                                                 const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char reason[ROCKE_ERR_MSG_CAP];
        if(b == NULL || spec == NULL)
            return NULL;
        rocke_wmma_fmha_decode_spec_t r;
        std::string op_id;
        bool gfx12 = false;
        if(!dec_valid(spec, arch, &r, &op_id, &gfx12, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "invalid wmma_fmha_decode spec for %s: %s", sz(arch), reason);
            return NULL;
        }
        if(dec_emit_body(b, &r, op_id.c_str(), gfx12) != ROCKE_OK)
            return NULL;
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_wmma_fmha_decode_merge(rocke_ir_builder_t* b,
                                                       const rocke_wmma_fmha_decode_spec_t* spec,
                                                       const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char reason[ROCKE_ERR_MSG_CAP];
        if(b == NULL || spec == NULL)
            return NULL;
        std::string op_id;
        bool gfx12 = false;
        if(!dec_post_init_ok(spec, reason, sizeof(reason))
           || !dec_layout(arch, spec, &op_id, &gfx12, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(b,
                                  ROCKE_ERR_VALUE,
                                  "invalid wmma_fmha_decode merge spec for %s: %s",
                                  sz(arch),
                                  reason);
            return NULL;
        }
        if(dec_emit_merge(b, spec) != ROCKE_OK)
            return NULL;
        return b->kernel;
    });
}

rocke_status_t rocke_wmma_fmha_decode_signature(const rocke_wmma_fmha_decode_spec_t* spec,
                                                rocke_arena_t* arena,
                                                const rocke_sig_entry_t** out_items,
                                                size_t* out_count)
{
    return dec_signature(spec, false, arena, out_items, out_count);
}

rocke_status_t rocke_wmma_fmha_decode_merge_signature(const rocke_wmma_fmha_decode_spec_t* spec,
                                                      rocke_arena_t* arena,
                                                      const rocke_sig_entry_t** out_items,
                                                      size_t* out_count)
{
    return dec_signature(spec, true, arena, out_items, out_count);
}

int rocke_wmma_fmha_decode_bucket(int head_dim_q, int head_dim_v)
{
    static const int buckets[3] = {64, 128, 256};
    static const int min_q[3] = {16, 72, 136}; /* _BUCKET_MIN */
    const int dq = head_dim_q, dv = head_dim_v;
    if(dq <= 0 || dv <= 0 || dq % 8 != 0 || dv % 8 != 0)
        return 0;
    const int widest = dq > dv ? dq : dv;
    for(int i = 0; i < 3; ++i)
        if(widest <= buckets[i])
            return dq >= min_q[i] ? buckets[i] : 0;
    return 0;
}

void rocke_wmma_fmha_decode_kv_range(int seqlen_q,
                                     int seqlen_k,
                                     bool bottom_right,
                                     int window_left,
                                     int window_right,
                                     int* lo,
                                     int* hi)
{
    const int64_t ctx = bottom_right ? (int64_t)seqlen_k - seqlen_q : 0;
    int64_t l = 0, h = seqlen_k;
    if(window_left >= 0)
    {
        l = ctx - window_left;
        if(l < 0)
            l = 0;
    }
    if(window_right >= 0)
    {
        h = ctx + window_right + seqlen_q;
        if(h > seqlen_k)
            h = seqlen_k;
        if(h < 0)
            h = 0;
    }
    if(lo != NULL)
        *lo = (int)l;
    if(hi != NULL)
        *hi = (int)h;
}

rocke_status_t rocke_wmma_fmha_decode_pack_groups(int num_query_heads,
                                                  int num_kv_heads,
                                                  int num_v_heads,
                                                  int* out)
{
    const int hq = num_query_heads, hk = num_kv_heads, hv = num_v_heads;
    if(out == NULL || hq <= 0 || hk <= 0 || hv <= 0 || hq % hk != 0 || hq % hv != 0)
        return ROCKE_ERR_VALUE;
    const int mx = hk > hv ? hk : hv;
    const int mn = hk > hv ? hv : hk;
    if(mx % mn != 0)
        return ROCKE_ERR_VALUE;
    *out = mx;
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_fmha_decode_row_tiles(
    int num_query_heads, int num_kv_heads, int num_v_heads, int seqlen_q, int* out)
{
    int groups = 0;
    if(out == NULL
       || rocke_wmma_fmha_decode_pack_groups(num_query_heads, num_kv_heads, num_v_heads, &groups)
              != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    const int64_t rows = (int64_t)(num_query_heads / groups) * seqlen_q;
    const int64_t tiles = (rows + DEC_ROWS - 1) / DEC_ROWS;
    if(tiles < 0 || tiles > 0x7FFFFFFF)
        return ROCKE_ERR_VALUE;
    *out = (int)tiles;
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_fmha_decode_num_splits(const rocke_wmma_fmha_decode_spec_t* spec,
                                                 int batch,
                                                 int pack_groups,
                                                 int row_tiles,
                                                 int kv_len,
                                                 int num_cus,
                                                 int wg_per_cu,
                                                 int min_tiles_per_wave,
                                                 int* out)
{
    if(spec == NULL || out == NULL || spec->dv_split <= 0 || spec->num_waves <= 0)
        return ROCKE_ERR_VALUE;
    if(wg_per_cu <= 0)
        wg_per_cu = 2;
    if(min_tiles_per_wave <= 0)
        min_tiles_per_wave = 4;
    int64_t groups = (int64_t)batch * pack_groups * row_tiles * spec->dv_split;
    if(groups < 1)
        groups = 1;
    const int64_t kl = kv_len > 0 ? kv_len : 0;
    const int64_t tiles = (kl + DEC_SPLIT_KEYS - 1) / DEC_SPLIT_KEYS;
    int64_t by_work = tiles / ((int64_t)spec->num_waves * min_tiles_per_wave);
    if(by_work < 1)
        by_work = 1;
    const int64_t cus = num_cus > 1 ? num_cus : 1;
    int64_t by_device = (cus * wg_per_cu) / groups;
    if(by_device < 1)
        by_device = 1;
    int64_t n = by_work < by_device ? by_work : by_device;
    if(n > DEC_MAX_SPLITS)
        n = DEC_MAX_SPLITS;
    *out = n < 1 ? 1 : (int)n;
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_fmha_decode_grid(const rocke_wmma_fmha_decode_spec_t* spec,
                                           int batch,
                                           int pack_groups,
                                           int row_tiles,
                                           int num_splits,
                                           int grid[3],
                                           int block[3])
{
    if(spec == NULL || grid == NULL || block == NULL || spec->dv_split <= 0)
        return ROCKE_ERR_VALUE;
    const int64_t n
        = (int64_t)batch * num_splits * pack_groups * spec->dv_split * (int64_t)row_tiles;
    if(!fits_i32(n))
        return ROCKE_ERR_VALUE;
    grid[0] = (int)n;
    grid[1] = 1;
    grid[2] = 1;
    block[0] = rocke_wmma_fmha_decode_block_size(spec);
    block[1] = 1;
    block[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_fmha_decode_merge_grid(
    int batch, int num_query_heads, int seqlen_q, int grid[3], int block[3])
{
    if(grid == NULL || block == NULL)
        return ROCKE_ERR_VALUE;
    const int64_t n = (int64_t)batch * num_query_heads * seqlen_q;
    if(!fits_i32(n))
        return ROCKE_ERR_VALUE;
    grid[0] = (int)n;
    grid[1] = 1;
    grid[2] = 1;
    block[0] = DEC_MERGE_THREADS;
    block[1] = 1;
    block[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_fmha_decode_workspace(const rocke_wmma_fmha_decode_spec_t* spec,
                                                int batch,
                                                int num_query_heads,
                                                int num_kv_heads,
                                                int num_v_heads,
                                                int seqlen_q,
                                                int head_dim_v,
                                                int num_splits,
                                                rocke_wmma_fmha_decode_workspace_t* out)
{
    if(spec == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    if(num_splits <= 1)
    {
        out->ws_o = out->ws_lse = out->ws_cnt = out->bytes = 0;
        return ROCKE_OK;
    }
    if(spec->dv_split <= 0)
        return ROCKE_ERR_VALUE;
    const int64_t rows = (int64_t)batch * num_query_heads * seqlen_q * num_splits;
    if(rows * ((int64_t)head_dim_v + 1) > 0x7FFFFFFF)
        return ROCKE_ERR_VALUE;
    int row_tiles = 0, groups = 0;
    if(rocke_wmma_fmha_decode_row_tiles(
           num_query_heads, num_kv_heads, num_v_heads, seqlen_q, &row_tiles)
           != ROCKE_OK
       || rocke_wmma_fmha_decode_pack_groups(num_query_heads, num_kv_heads, num_v_heads, &groups)
              != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    const int64_t lse = rows * head_dim_v * 4;
    const int64_t cnt = lse + rows * 4;
    const int64_t counters = (int64_t)batch * groups * spec->dv_split * row_tiles;
    out->ws_o = 0;
    out->ws_lse = lse;
    out->ws_cnt = cnt;
    out->bytes = cnt + counters * 4;
    return ROCKE_OK;
}

bool rocke_wmma_fmha_decode_fused_merge(int num_splits)
{
    return 1 < num_splits && num_splits <= DEC_FUSED_MAX_SPLITS;
}

rocke_status_t rocke_wmma_fmha_decode_lower_to_llvm(const rocke_wmma_fmha_decode_spec_t* spec,
                                                    bool merge,
                                                    const char* arch,
                                                    rocke_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap)
{
    char name_buf[512];
    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL || arch == NULL)
    {
        rocke_spec_set_reason(err, err_cap, "lower_to_llvm: null spec/arch/out");
        return ROCKE_ERR_VALUE;
    }
    rocke_wmma_fmha_decode_spec_t r = *spec;
    rocke_status_t st;
    if(merge)
        st = rocke_wmma_fmha_decode_merge_kernel_name(spec, name_buf, sizeof(name_buf));
    else
    {
        st = rocke_wmma_fmha_decode_resolve(spec, arch, &r, err, err_cap);
        if(st == ROCKE_OK)
            st = rocke_wmma_fmha_decode_kernel_name(&r, name_buf, sizeof(name_buf));
    }
    if(st != ROCKE_OK)
    {
        if(err != NULL && err_cap > 0 && err[0] == '\0')
            rocke_spec_set_reason(err, err_cap, "lower_to_llvm: invalid spec or name too long");
        return ROCKE_ERR_VALUE;
    }
    rocke_ir_builder_t b;
    st = rocke_ir_builder_init(&b, name_buf);
    if(st != ROCKE_OK)
    {
        rocke_spec_set_reason(err, err_cap, "lower_to_llvm: builder init failed");
        return st;
    }
    rocke_kernel_def_t* k = merge ? rocke_build_wmma_fmha_decode_merge(&b, spec, arch)
                                  : rocke_build_wmma_fmha_decode(&b, &r, arch);
    if(k == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        rocke_spec_set_reason(err, err_cap, m != NULL ? m : "build_wmma_fmha_decode failed");
        st = rocke_ir_builder_status(&b);
        rocke_ir_builder_free(&b);
        return st != ROCKE_OK ? st : ROCKE_ERR_VALUE;
    }
    st = rocke_lower_kernel_to_llvm_ex(k, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
