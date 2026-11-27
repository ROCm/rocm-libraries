// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * batchnorm2d_mcw.cpp -- C99 port of the multi-channel-per-wave kernels of
 * rocke/instances/common/batchnorm2d.py (BatchNorm2DMcwSpec,
 * build_batchnorm2d_mcw_train / build_batchnorm2d_mcw_bwd).
 *
 * Byte-faithful reproduction of the Python builder-call sequence: every
 * nested Python call is unrolled into its own statement in Python's
 * left-to-right evaluation order (C++ leaves argument evaluation order
 * unspecified, and the order fixes the SSA numbering).
 */

#include "rocke/instance_batchnorm2d_mcw.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/instance_batchnorm2d_bwd_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

rocke_batchnorm2d_mcw_spec_t rocke_batchnorm2d_mcw_spec_default(void)
{
    rocke_batchnorm2d_mcw_spec_t s;
    s.mode = "train";
    s.waves = 4;
    s.cpw = 1;
    s.npw = 32;
    s.slots = 1;
    s.vec = 1;
    s.dtype = "f16";
    s.wave_size = 64;
    s.dynamic = true;
    s.name = "rocke_batchnorm2d_mcw";
    return s;
}

int rocke_batchnorm2d_mcw_block_size(const rocke_batchnorm2d_mcw_spec_t* spec)
{
    return spec ? spec->waves * spec->wave_size : 0;
}

int rocke_batchnorm2d_mcw_lpc(const rocke_batchnorm2d_mcw_spec_t* spec)
{
    return (spec && spec->cpw > 0) ? spec->wave_size / spec->cpw : 0;
}

int rocke_batchnorm2d_mcw_regs(const rocke_batchnorm2d_mcw_spec_t* spec)
{
    if(spec == NULL)
        return 0;
    return spec->npw * spec->slots * spec->vec * (strcmp(spec->mode, "train") == 0 ? 1 : 2);
}

/* kernel_name_join(name, mode, dtype, f"w{waves}", f"cp{cpw}", f"n{npw}",
 *                  f"s{slots}", f"v{vec}" if vec != 1 else "") */
rocke_status_t rocke_batchnorm2d_mcw_kernel_name(const rocke_batchnorm2d_mcw_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char pw[32], pcp[32], pn[32], ps[32], pv[32];
    const char* parts[7];
    if(spec == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    snprintf(pw, sizeof pw, "w%d", spec->waves);
    snprintf(pcp, sizeof pcp, "cp%d", spec->cpw);
    snprintf(pn, sizeof pn, "n%d", spec->npw);
    snprintf(ps, sizeof ps, "s%d", spec->slots);
    if(spec->vec != 1)
        snprintf(pv, sizeof pv, "v%d", spec->vec);
    else
        pv[0] = '\0';
    parts[0] = spec->mode;
    parts[1] = spec->dtype;
    parts[2] = pw;
    parts[3] = pcp;
    parts[4] = pn;
    parts[5] = ps;
    parts[6] = pv;
    return rocke_kernel_name_join(spec->name, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
}

static bool mcw_reject(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
    return false;
}

bool rocke_batchnorm2d_is_valid_mcw_spec(const rocke_batchnorm2d_mcw_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    const rocke_archtarget_t* target;
    char msg[128];
    bool f32;
    if(reason != NULL && reason_cap > 0)
        reason[0] = '\0';
    if(spec == NULL)
        return mcw_reject(reason, reason_cap, "null spec");
    if(arch == NULL)
        arch = "gfx950";
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
        return mcw_reject(reason, reason_cap, "unknown arch");
    if(spec->mode == NULL || (strcmp(spec->mode, "train") && strcmp(spec->mode, "bwd")))
    {
        snprintf(msg, sizeof msg, "unknown mode '%s'", spec->mode ? spec->mode : "?");
        return mcw_reject(reason, reason_cap, msg);
    }
    if(spec->dtype == NULL
       || (strcmp(spec->dtype, "f16") && strcmp(spec->dtype, "bf16") && strcmp(spec->dtype, "f32")))
    {
        snprintf(msg, sizeof msg, "unsupported dtype '%s'", spec->dtype ? spec->dtype : "?");
        return mcw_reject(reason, reason_cap, msg);
    }
    if(spec->cpw <= 0 || spec->cpw > 64 || (spec->cpw & (spec->cpw - 1)))
        return mcw_reject(reason, reason_cap, "cpw must be a power of two dividing the wave");
    if(spec->waves <= 0 || (spec->waves & (spec->waves - 1)))
        return mcw_reject(reason, reason_cap, "waves must be a power of two");
    f32 = strcmp(spec->dtype, "f32") == 0;
    if(!(spec->vec == 1 || spec->vec == 2 || spec->vec == 4 || (!f32 && spec->vec == 8)))
        return mcw_reject(reason, reason_cap, "vec out of range");
    if(spec->npw <= 0 || spec->slots <= 0)
        return mcw_reject(reason, reason_cap, "npw / slots must be positive");
    if(rocke_batchnorm2d_mcw_regs(spec) > ROCKE_BATCHNORM2D_MCW_MAX_REGS)
        return mcw_reject(reason, reason_cap, "register-cached elements per lane out of budget");
    if(rocke_batchnorm2d_mcw_block_size(spec) > rocke_archtarget_max_threads_per_block(target))
        return mcw_reject(reason, reason_cap, "block_size exceeds max_threads_per_block");
    if(!rocke_archtarget_fits_lds(target, 2L * spec->waves * spec->cpw * 4))
        return mcw_reject(reason, reason_cap, "LDS budget exceeded");
    return true;
}

static int mcw_ceil_pow2(int v)
{
    int p = 1;
    while(p < v)
        p *= 2;
    return p;
}

bool rocke_batchnorm2d_mcw_admits(const rocke_batchnorm2d_mcw_spec_t* spec,
                                  int n,
                                  int c,
                                  int h,
                                  int w,
                                  char* reason,
                                  size_t reason_cap)
{
    long long hw = (long long)h * w;
    if(reason != NULL && reason_cap > 0)
        reason[0] = '\0';
    if(spec == NULL)
        return mcw_reject(reason, reason_cap, "null spec");
    if(n <= 0 || c <= 0 || hw <= 0)
        return mcw_reject(reason, reason_cap, "empty tensor");
    if(c % spec->cpw != 0)
        return mcw_reject(reason, reason_cap, "C not a multiple of cpw");
    if(spec->waves > n)
        return mcw_reject(reason, reason_cap, "more waves than samples");
    if(spec->npw != mcw_ceil_pow2((n + spec->waves - 1) / spec->waves))
        return mcw_reject(reason, reason_cap, "npw does not match the sample count");
    if(hw % spec->vec != 0)
        return mcw_reject(reason, reason_cap, "spatial size not a multiple of vec");
    {
        long long lpc = rocke_batchnorm2d_mcw_lpc(spec);
        long long per = hw / spec->vec;
        if(spec->slots != mcw_ceil_pow2((int)((per + lpc - 1) / lpc)))
            return mcw_reject(reason, reason_cap, "slots do not match the spatial size");
    }
    if((long long)n * c * hw + rocke_batchnorm2d_mcw_block_size(spec) >= (1LL << 31))
        return mcw_reject(reason, reason_cap, "element count out of i32 range");
    return true;
}

rocke_status_t
    rocke_batchnorm2d_mcw_grid(int c, const rocke_batchnorm2d_mcw_spec_t* spec, int out[3])
{
    if(out == NULL || spec == NULL || spec->cpw <= 0)
        return ROCKE_ERR_VALUE;
    out[0] = c / spec->cpw;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * shared pieces
 * ------------------------------------------------------------------ */

static rocke_value_t*
    mcw_fadd_combine(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, void* user)
{
    (void)user;
    return rocke_b_fadd(b, a, c);
}

static rocke_param_opts_t mcw_opts(bool readonly, bool writeonly, int align)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof o);
    o.noalias = true;
    o.noalias_set = true;
    if(readonly)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    if(writeonly)
    {
        o.writeonly = true;
        o.writeonly_set = true;
    }
    if(align > 0)
    {
        o.align = align;
        o.align_set = true;
    }
    return o;
}

static rocke_value_t* mcw_param(rocke_ir_builder_t* b,
                                const char* name,
                                const rocke_type_t* elem,
                                bool readonly,
                                bool writeonly,
                                int align)
{
    rocke_param_opts_t o = mcw_opts(readonly, writeonly, align);
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &o);
}

/* Python _McwCtx: lane coordinates and cached-element addressing. */
typedef struct mcw_ctx
{
    rocke_value_t *p_n, *p_c, *hw, *tid, *wave, *lane, *cj, *p, *channel, *owner;
    rocke_value_t** hw_idx;
    rocke_value_t** hw_ok;
    rocke_value_t** n_idx;
    rocke_value_t** n_ok;
} mcw_ctx_t;

static rocke_value_t** mcw_alloc_vals(rocke_ir_builder_t* b, int n)
{
    rocke_value_t** v = (rocke_value_t**)rocke_arena_alloc(
        &b->arena, sizeof(rocke_value_t*) * (size_t)(n > 0 ? n : 1));
    if(v == NULL)
        rocke_i_set_err(b, ROCKE_ERR_OOM, "batchnorm2d_mcw: OOM");
    return v;
}

static void mcw_ctx_init(rocke_ir_builder_t* b,
                         const rocke_batchnorm2d_mcw_spec_t* spec,
                         mcw_ctx_t* ctx,
                         rocke_value_t* p_n,
                         rocke_value_t* p_c,
                         rocke_value_t* p_h,
                         rocke_value_t* p_w)
{
    const int W = spec->waves, CPW = spec->cpw, LPC = rocke_batchnorm2d_mcw_lpc(spec);
    rocke_value_t *wsz, *c, *bid, *t, *hw_last, *n_last;
    ctx->p_n = p_n;
    ctx->p_c = p_c;
    ctx->hw = rocke_b_mul(b, p_h, p_w);
    ctx->tid = rocke_b_thread_id_x(b);
    wsz = rocke_b_const_i32(b, spec->wave_size);
    ctx->wave = rocke_b_div(b, ctx->tid, wsz);
    ctx->lane = rocke_b_mod(b, ctx->tid, wsz);
    c = rocke_b_const_i32(b, LPC);
    ctx->cj = rocke_b_div(b, ctx->lane, c);
    c = rocke_b_const_i32(b, LPC);
    ctx->p = rocke_b_mod(b, ctx->lane, c);
    bid = rocke_b_block_id_x(b);
    c = rocke_b_const_i32(b, CPW);
    t = rocke_b_mul(b, bid, c);
    ctx->channel = rocke_b_add(b, t, ctx->cj);
    c = rocke_b_const_i32(b, spec->vec);
    t = rocke_b_div(b, ctx->hw, c);
    c = rocke_b_const_i32(b, 1);
    hw_last = rocke_b_sub(b, t, c);
    ctx->hw_idx = mcw_alloc_vals(b, spec->slots);
    ctx->hw_ok = mcw_alloc_vals(b, spec->slots);
    for(int s = 0; s < spec->slots && ctx->hw_ok; ++s)
    {
        c = rocke_b_const_i32(b, LPC * s);
        rocke_value_t* hs = rocke_b_add(b, ctx->p, c);
        ctx->hw_ok[s] = rocke_b_cmp_le(b, hs, hw_last);
        ctx->hw_idx[s] = rocke_b_smin(b, hs, hw_last);
    }
    c = rocke_b_const_i32(b, 1);
    n_last = rocke_b_sub(b, p_n, c);
    ctx->n_idx = mcw_alloc_vals(b, spec->npw);
    ctx->n_ok = mcw_alloc_vals(b, spec->npw);
    for(int k = 0; k < spec->npw && ctx->n_ok; ++k)
    {
        c = rocke_b_const_i32(b, k * W);
        rocke_value_t* nk = rocke_b_add(b, c, ctx->wave);
        ctx->n_ok[k] = rocke_b_cmp_lt(b, nk, p_n);
        ctx->n_idx[k] = rocke_b_smin(b, nk, n_last);
    }
    {
        rocke_value_t* z = rocke_b_const_i32(b, 0);
        rocke_value_t* e1 = rocke_b_cmp_eq(b, ctx->wave, z);
        z = rocke_b_const_i32(b, 0);
        rocke_value_t* e2 = rocke_b_cmp_eq(b, ctx->p, z);
        ctx->owner = rocke_b_land(b, e1, e2);
    }
}

/* _McwCtx.elems(): (flat index, valid) of every cached element, sample-major. */
static int mcw_elems(rocke_ir_builder_t* b,
                     const rocke_batchnorm2d_mcw_spec_t* spec,
                     const mcw_ctx_t* ctx,
                     rocke_value_t** idx,
                     rocke_value_t** ok)
{
    int e = 0;
    for(int k = 0; k < spec->npw; ++k)
    {
        rocke_value_t* t = rocke_b_mul(b, ctx->n_idx[k], ctx->p_c);
        t = rocke_b_add(b, t, ctx->channel);
        rocke_value_t* base = rocke_b_mul(b, t, ctx->hw);
        for(int s = 0; s < spec->slots; ++s)
        {
            rocke_value_t* col = ctx->hw_idx[s];
            if(spec->vec != 1)
            {
                rocke_value_t* c = rocke_b_const_i32(b, spec->vec);
                col = rocke_b_mul(b, ctx->hw_idx[s], c);
            }
            idx[e] = rocke_b_add(b, base, col);
            ok[e] = rocke_b_land(b, ctx->n_ok[k], ctx->hw_ok[s]);
            ++e;
        }
    }
    return e;
}

/* _mcw_load: the `vec` consecutive elements at idx as f32 scalars (into out). */
static void mcw_load(rocke_ir_builder_t* b,
                     const rocke_batchnorm2d_mcw_spec_t* spec,
                     rocke_value_t* ptr,
                     rocke_value_t* idx,
                     rocke_value_t** out)
{
    if(spec->vec == 1)
        out[0] = rocke_b_load_scalar_as_f32(b, ptr, idx, spec->dtype);
    else
        (void)rocke_b_load_vec_as_f32(b, ptr, idx, spec->dtype, spec->vec, out);
}

static void mcw_store(rocke_ir_builder_t* b,
                      const rocke_batchnorm2d_mcw_spec_t* spec,
                      rocke_value_t* ptr,
                      rocke_value_t* idx,
                      rocke_value_t* const* vals)
{
    if(spec->vec == 1)
    {
        rocke_b_store_scalar_from_f32(b, ptr, idx, vals[0], spec->dtype);
    }
    else
    {
        rocke_value_t* packed = rocke_b_pack_f32_to(b, vals, spec->vec, spec->dtype);
        rocke_b_store_vec(b, ptr, idx, packed, spec->vec);
    }
}

/* _mcw_group_sums: sum each of the two per-lane partials over its channel's
 * lanes and waves; every lane gets the totals of its own channel. */
static void mcw_group_sums(rocke_ir_builder_t* b,
                           const rocke_batchnorm2d_mcw_spec_t* spec,
                           const mcw_ctx_t* ctx,
                           rocke_value_t* const* vals,
                           int nv,
                           rocke_value_t* lds,
                           rocke_value_t** out)
{
    const int W = spec->waves, CPW = spec->cpw, LPC = rocke_batchnorm2d_mcw_lpc(spec);
    rocke_value_t* tot[2];
    for(int q = 0; q < nv; ++q)
        tot[q] = rocke_warp_xor_reduce(b, vals[q], ROCKE_REDUCE_SUM, LPC);
    if(W == 1)
    {
        for(int q = 0; q < nv; ++q)
            out[q] = tot[q];
        return;
    }
    {
        rocke_value_t* z = rocke_b_const_i32(b, 0);
        rocke_value_t* cond = rocke_b_cmp_eq(b, ctx->p, z);
        rocke_if_t iff = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, iff.then_region);
        for(int q = 0; q < nv; ++q)
        {
            rocke_value_t* c = rocke_b_const_i32(b, CPW);
            rocke_value_t* t = rocke_b_mul(b, ctx->wave, c);
            rocke_value_t* slot = rocke_b_add(b, t, ctx->cj);
            c = rocke_b_const_i32(b, q * W * CPW);
            rocke_value_t* at = rocke_b_add(b, slot, c);
            rocke_b_smem_store_vN_f32(b, lds, &at, 1, tot[q], 1);
        }
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);
    for(int q = 0; q < nv; ++q)
    {
        rocke_value_t* parts[64];
        for(int w = 0; w < W; ++w)
        {
            rocke_value_t* c = rocke_b_const_i32(b, q * W * CPW + w * CPW);
            rocke_value_t* slot = rocke_b_add(b, ctx->cj, c);
            rocke_value_t* v = rocke_b_smem_load_vN_f32(b, lds, &slot, 1, 1);
            parts[w] = rocke_b_vec_extract(b, v, 0);
        }
        out[q] = rocke_tree_reduce(b, mcw_fadd_combine, NULL, parts, W);
    }
}

static rocke_value_t*
    mcw_lds(rocke_ir_builder_t* b, const rocke_batchnorm2d_mcw_spec_t* spec, int n_vals)
{
    rocke_tensor_view_t v;
    int shape[1];
    shape[0] = n_vals * spec->waves * spec->cpw;
    if(rocke_make_lds_view(b, &v, rocke_f32(), shape, 1, "lds_mcw", NULL) != ROCKE_OK)
        return NULL;
    return v.base;
}

static bool
    mcw_check(rocke_ir_builder_t* b, const rocke_batchnorm2d_mcw_spec_t* spec, const char* mode)
{
    char why[256];
    if(!rocke_batchnorm2d_is_valid_mcw_spec(spec, NULL, why, sizeof why)
       || strcmp(spec->mode, mode))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "invalid mcw %s spec: %s",
                        strcmp(mode, "train") == 0 ? "training" : "backward",
                        why[0] ? why : spec->mode);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ *
 * build_batchnorm2d_mcw_train
 * ------------------------------------------------------------------ */

rocke_kernel_def_t* rocke_build_batchnorm2d_mcw_train(rocke_ir_builder_t* b,
                                                      const rocke_batchnorm2d_mcw_spec_t* spec)
{
    if(b == NULL || spec == NULL || !mcw_check(b, spec, "train"))
        return NULL;
    const rocke_type_t* io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
        return NULL;
    const int VEC = spec->vec;
    const int NE = spec->npw * spec->slots;
    rocke_attr_set_int(
        b, &b->kernel->attrs, "max_workgroup_size", rocke_batchnorm2d_mcw_block_size(spec));
    rocke_value_t* X = mcw_param(b, "X", io_ty, true, false, 16);
    rocke_value_t* Gamma = mcw_param(b, "Gamma", io_ty, true, false, 16);
    rocke_value_t* Beta = mcw_param(b, "Beta", io_ty, true, false, 16);
    rocke_value_t* Y = mcw_param(b, "Y", io_ty, false, true, 16);
    rocke_value_t* RunningMean = mcw_param(b, "RunningMean", rocke_f32(), false, false, 0);
    rocke_value_t* RunningVar = mcw_param(b, "RunningVar", rocke_f32(), false, false, 0);
    rocke_value_t* Mean = mcw_param(b, "Mean", io_ty, false, true, 0);
    rocke_value_t* InvStd = mcw_param(b, "InvStd", io_ty, false, true, 0);
    rocke_value_t* p_n = rocke_b_param(b, "N", rocke_i32(), NULL);
    rocke_value_t* p_c = rocke_b_param(b, "C", rocke_i32(), NULL);
    rocke_value_t* p_h = rocke_b_param(b, "H", rocke_i32(), NULL);
    rocke_value_t* p_w = rocke_b_param(b, "W", rocke_i32(), NULL);
    rocke_value_t* eps = rocke_b_param(b, "eps", rocke_f32(), NULL);
    rocke_value_t* momentum = rocke_b_param(b, "momentum", rocke_f32(), NULL);

    mcw_ctx_t ctx;
    mcw_ctx_init(b, spec, &ctx, p_n, p_c, p_h, p_w);
    rocke_value_t* lds = mcw_lds(b, spec, 2);
    rocke_value_t* t = rocke_b_mul(b, ctx.channel, ctx.hw);
    rocke_value_t* shift = rocke_b_load_scalar_as_f32(b, X, t, spec->dtype);
    rocke_value_t* gamma = rocke_b_load_scalar_as_f32(b, Gamma, ctx.channel, spec->dtype);
    rocke_value_t* beta = rocke_b_load_scalar_as_f32(b, Beta, ctx.channel, spec->dtype);
    rocke_value_t** idx = mcw_alloc_vals(b, NE);
    rocke_value_t** ok = mcw_alloc_vals(b, NE);
    rocke_value_t** xs = mcw_alloc_vals(b, NE * VEC);
    rocke_value_t** ds = mcw_alloc_vals(b, NE * VEC);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    mcw_elems(b, spec, &ctx, idx, ok);
    for(int e = 0; e < NE; ++e)
        mcw_load(b, spec, X, idx[e], xs + e * VEC);
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    for(int e = 0; e < NE; ++e)
    {
        for(int i = 0; i < VEC; ++i)
        {
            rocke_value_t* d = rocke_b_fsub(b, xs[e * VEC + i], shift);
            ds[e * VEC + i] = rocke_b_select(b, ok[e], d, zero);
        }
    }
    rocke_value_t* s1 = rocke_tree_reduce(b, mcw_fadd_combine, NULL, ds, NE * VEC);
    {
        rocke_value_t** sq = mcw_alloc_vals(b, NE * VEC);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        for(int i = 0; i < NE * VEC; ++i)
            sq[i] = rocke_b_fmul(b, ds[i], ds[i]);
        rocke_value_t* s2 = rocke_tree_reduce(b, mcw_fadd_combine, NULL, sq, NE * VEC);
        rocke_value_t* sums[2] = {s1, s2};
        rocke_value_t* tot[2] = {NULL, NULL};
        mcw_group_sums(b, spec, &ctx, sums, 2, lds, tot);
        s1 = tot[0];
        s2 = tot[1];

        rocke_value_t* nhw = rocke_b_mul(b, p_n, ctx.hw);
        rocke_value_t* m_f = rocke_b_sitofp_f32(b, nhw);
        rocke_value_t* one = rocke_b_const_f32(b, 1.0);
        rocke_value_t* inv_m = rocke_b_fdiv(b, one, m_f);
        rocke_value_t* ms = rocke_b_fmul(b, s1, inv_m);
        rocke_value_t* a = rocke_b_fmul(b, s2, inv_m);
        rocke_value_t* m2 = rocke_b_fmul(b, ms, ms);
        rocke_value_t* dv = rocke_b_fsub(b, a, m2);
        rocke_value_t* var = rocke_b_fmax(b, dv, zero);
        rocke_value_t* mean = rocke_b_fadd(b, shift, ms);
        rocke_value_t* ve = rocke_b_fadd(b, var, eps);
        rocke_value_t* inv = rocke_b_rsqrt(b, ve);
        {
            rocke_if_t iff = rocke_b_scf_if(b, ctx.owner);
            rocke_b_region_enter(b, iff.then_region);
            rocke_value_t* c1 = rocke_b_const_f32(b, 1.0);
            rocke_value_t* one_minus = rocke_b_fsub(b, c1, momentum);
            c1 = rocke_b_const_f32(b, 1.0);
            rocke_value_t* mm1 = rocke_b_fsub(b, m_f, c1);
            c1 = rocke_b_const_f32(b, 1.0);
            rocke_value_t* den = rocke_b_fmax(b, mm1, c1);
            rocke_value_t* unbias = rocke_b_fdiv(b, m_f, den);
            rocke_value_t* old_m = rocke_b_global_load_f32(b, RunningMean, ctx.channel, 4);
            rocke_value_t* old_v = rocke_b_global_load_f32(b, RunningVar, ctx.channel, 4);
            rocke_value_t* u1 = rocke_b_fmul(b, one_minus, old_m);
            rocke_value_t* u2 = rocke_b_fmul(b, momentum, mean);
            rocke_b_global_store(b, RunningMean, ctx.channel, rocke_b_fadd(b, u1, u2), 1);
            rocke_value_t* v1 = rocke_b_fmul(b, one_minus, old_v);
            rocke_value_t* v2 = rocke_b_fmul(b, var, unbias);
            rocke_value_t* v3 = rocke_b_fmul(b, momentum, v2);
            rocke_b_global_store(b, RunningVar, ctx.channel, rocke_b_fadd(b, v1, v3), 1);
            rocke_b_store_scalar_from_f32(b, Mean, ctx.channel, mean, spec->dtype);
            rocke_b_store_scalar_from_f32(b, InvStd, ctx.channel, inv, spec->dtype);
            rocke_b_region_leave(b);
        }
        rocke_value_t* sc = rocke_b_fmul(b, inv, gamma);
        rocke_value_t* msc = rocke_b_fmul(b, mean, sc);
        rocke_value_t* sh = rocke_b_fsub(b, beta, msc);
        for(int e = 0; e < NE; ++e)
        {
            rocke_if_t iff = rocke_b_scf_if(b, ok[e]);
            rocke_b_region_enter(b, iff.then_region);
            rocke_value_t* ys[8];
            for(int i = 0; i < VEC; ++i)
                ys[i] = rocke_b_fma(b, xs[e * VEC + i], sc, sh);
            mcw_store(b, spec, Y, idx[e], ys);
            rocke_b_region_leave(b);
        }
    }
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

/* ------------------------------------------------------------------ *
 * build_batchnorm2d_mcw_bwd
 * ------------------------------------------------------------------ */

rocke_kernel_def_t* rocke_build_batchnorm2d_mcw_bwd(rocke_ir_builder_t* b,
                                                    const rocke_batchnorm2d_mcw_spec_t* spec)
{
    if(b == NULL || spec == NULL || !mcw_check(b, spec, "bwd"))
        return NULL;
    const rocke_type_t* io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
        return NULL;
    const int VEC = spec->vec;
    const int NE = spec->npw * spec->slots;
    rocke_attr_set_int(
        b, &b->kernel->attrs, "max_workgroup_size", rocke_batchnorm2d_mcw_block_size(spec));
    rocke_value_t* X = mcw_param(b, "X", io_ty, true, false, 16);
    rocke_value_t* dY = mcw_param(b, "dY", io_ty, true, false, 16);
    rocke_value_t* Scale = mcw_param(b, "Scale", rocke_f32(), true, false, 0);
    rocke_value_t* Mean = mcw_param(b, "Mean", rocke_f32(), true, false, 0);
    rocke_value_t* InvStd = mcw_param(b, "InvStd", rocke_f32(), true, false, 0);
    rocke_value_t* dX = mcw_param(b, "dX", io_ty, false, true, 16);
    rocke_value_t* dScale = mcw_param(b, "dScale", rocke_f32(), false, true, 0);
    rocke_value_t* dBias = mcw_param(b, "dBias", rocke_f32(), false, true, 0);
    rocke_value_t* p_n = rocke_b_param(b, "N", rocke_i32(), NULL);
    rocke_value_t* p_c = rocke_b_param(b, "C", rocke_i32(), NULL);
    rocke_value_t* p_h = rocke_b_param(b, "H", rocke_i32(), NULL);
    rocke_value_t* p_w = rocke_b_param(b, "W", rocke_i32(), NULL);

    mcw_ctx_t ctx;
    mcw_ctx_init(b, spec, &ctx, p_n, p_c, p_h, p_w);
    rocke_value_t* lds = mcw_lds(b, spec, 2);
    rocke_value_t* mean = rocke_b_global_load_f32(b, Mean, ctx.channel, 4);
    rocke_value_t* invstd = rocke_b_global_load_f32(b, InvStd, ctx.channel, 4);
    rocke_value_t* scale = rocke_b_global_load_f32(b, Scale, ctx.channel, 4);
    rocke_value_t** idx = mcw_alloc_vals(b, NE);
    rocke_value_t** ok = mcw_alloc_vals(b, NE);
    rocke_value_t** xs = mcw_alloc_vals(b, NE * VEC);
    rocke_value_t** dys = mcw_alloc_vals(b, NE * VEC);
    rocke_value_t** dym = mcw_alloc_vals(b, NE * VEC);
    rocke_value_t** prods = mcw_alloc_vals(b, NE * VEC);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    mcw_elems(b, spec, &ctx, idx, ok);
    for(int e = 0; e < NE; ++e)
        mcw_load(b, spec, X, idx[e], xs + e * VEC);
    for(int e = 0; e < NE; ++e)
        mcw_load(b, spec, dY, idx[e], dys + e * VEC);
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    for(int e = 0; e < NE; ++e)
        for(int i = 0; i < VEC; ++i)
            dym[e * VEC + i] = rocke_b_select(b, ok[e], dys[e * VEC + i], zero);
    for(int i = 0; i < NE * VEC; ++i)
    {
        rocke_value_t* d = rocke_b_fsub(b, xs[i], mean);
        rocke_value_t* xh = rocke_b_fmul(b, d, invstd);
        prods[i] = rocke_b_fmul(b, dym[i], xh);
    }
    rocke_value_t* s1 = rocke_tree_reduce(b, mcw_fadd_combine, NULL, dym, NE * VEC);
    rocke_value_t* s2 = rocke_tree_reduce(b, mcw_fadd_combine, NULL, prods, NE * VEC);
    rocke_value_t* sums[2] = {s1, s2};
    rocke_value_t* tot[2] = {NULL, NULL};
    mcw_group_sums(b, spec, &ctx, sums, 2, lds, tot);
    {
        rocke_if_t iff = rocke_b_scf_if(b, ctx.owner);
        rocke_b_region_enter(b, iff.then_region);
        rocke_b_global_store(b, dScale, ctx.channel, tot[1], 1);
        rocke_b_global_store(b, dBias, ctx.channel, tot[0], 1);
        rocke_b_region_leave(b);
    }
    rocke_value_t* nhw = rocke_b_mul(b, p_n, ctx.hw);
    rocke_value_t* m_f = rocke_b_sitofp_f32(b, nhw);
    rocke_value_t *a, *k2, *k3, *coefs[3];
    rocke_bn_bwd_coefs(b, tot[0], tot[1], scale, mean, invstd, m_f, coefs);
    a = coefs[0];
    k2 = coefs[1];
    k3 = coefs[2];
    for(int e = 0; e < NE; ++e)
    {
        rocke_if_t iff = rocke_b_scf_if(b, ok[e]);
        rocke_b_region_enter(b, iff.then_region);
        rocke_value_t* dx[8];
        for(int i = 0; i < VEC; ++i)
        {
            rocke_value_t* ady = rocke_b_fmul(b, a, dys[e * VEC + i]);
            rocke_value_t* xc = rocke_b_fsub(b, xs[e * VEC + i], mean);
            rocke_value_t* kx = rocke_b_fmul(b, k2, xc);
            rocke_value_t* u = rocke_b_fsub(b, ady, kx);
            dx[i] = rocke_b_fsub(b, u, k3);
        }
        mcw_store(b, spec, dX, idx[e], dx);
        rocke_b_region_leave(b);
    }
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_batchnorm2d_mcw(rocke_ir_builder_t* b,
                                                const rocke_batchnorm2d_mcw_spec_t* spec)
{
    if(spec == NULL)
        return NULL;
    return strcmp(spec->mode, "train") == 0 ? rocke_build_batchnorm2d_mcw_train(b, spec)
                                            : rocke_build_batchnorm2d_mcw_bwd(b, spec);
}

rocke_kernel_def_t* rocke_build_batchnorm2d_mcw_new(rocke_ir_builder_t* b,
                                                    const rocke_batchnorm2d_mcw_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
            return NULL;
        if(rocke_batchnorm2d_mcw_kernel_name(spec, name, sizeof name) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_batchnorm2d_mcw(b, spec);
    });
}
