// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * batchnorm2d_nhwc.cpp -- C99 port of the dynamic NHWC BatchNorm2D kernels of
 * rocke/instances/common/batchnorm2d.py: build_batchnorm2d_nhwc_infer,
 * build_batchnorm2d_nhwc_train and build_batchnorm2d_nhwc_bwd.
 *
 * Byte-faithful reproduction of the Python builder-call sequence: every
 * builder call that emits IR is issued in Python's left-to-right evaluation
 * order (no two emitting calls share one C argument list).
 */

#include "rocke/instance_batchnorm2d_nhwc.h"

#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/instance_batchnorm2d_bwd_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#define NHWC_MAX_VEC 8
#define NHWC_MAX_UNROLL 16

/* ------------------------------------------------------------------ *
 * spec defaults / properties / name / validity
 * ------------------------------------------------------------------ */

rocke_batchnorm2d_nhwc_spec_t rocke_batchnorm2d_nhwc_spec_default(void)
{
    rocke_batchnorm2d_nhwc_spec_t s;
    s.mode = "train";
    s.block_size = 256;
    s.tx = 8;
    s.vec = 8;
    s.unroll = 4;
    s.dtype = "f16";
    s.wave_size = 64;
    s.dynamic = true;
    s.name = "rocke_batchnorm2d_nhwc";
    return s;
}

int rocke_batchnorm2d_nhwc_ty(const rocke_batchnorm2d_nhwc_spec_t* spec)
{
    if(spec == NULL || spec->tx == 0)
    {
        return 0;
    }
    return spec->block_size / spec->tx;
}

int rocke_batchnorm2d_nhwc_tile_channels(const rocke_batchnorm2d_nhwc_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->tx * spec->vec;
}

rocke_status_t rocke_batchnorm2d_nhwc_kernel_name(const rocke_batchnorm2d_nhwc_spec_t* spec,
                                                  const char* kind,
                                                  char* out,
                                                  size_t out_cap)
{
    char part_b[32], part_tx[32], part_v[32], part_u[32];
    const char* parts[7];

    if(spec == NULL || kind == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(part_b, sizeof part_b, "b%d", spec->block_size);
    snprintf(part_tx, sizeof part_tx, "tx%d", spec->tx);
    snprintf(part_v, sizeof part_v, "v%d", spec->vec);
    snprintf(part_u, sizeof part_u, "u%d", spec->unroll);
    parts[0] = spec->mode;
    parts[1] = kind;
    parts[2] = spec->dtype;
    parts[3] = part_b;
    parts[4] = part_tx;
    parts[5] = part_v;
    parts[6] = part_u;
    return rocke_kernel_name_join(spec->name, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
}

static bool nhwc_is_pow2(int v)
{
    return v > 0 && (v & (v - 1)) == 0;
}

bool rocke_batchnorm2d_is_valid_nhwc_spec(const rocke_batchnorm2d_nhwc_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap)
{
    const rocke_archtarget_t* target;
    char msg[96];
    int ty;

    if(reason != NULL && reason_cap > 0)
    {
        reason[0] = '\0';
    }
    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(spec->wave_size <= 0 || (spec->wave_size & (spec->wave_size - 1)) != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "wave_size must be a positive power of two");
        return false;
    }
    target = rocke_archtarget_from_gfx(arch != NULL ? arch : "gfx950");
    if(target == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "unknown arch");
        return false;
    }
    if(spec->mode == NULL
       || (strcmp(spec->mode, "infer") != 0 && strcmp(spec->mode, "train") != 0
           && strcmp(spec->mode, "bwd") != 0))
    {
        snprintf(msg, sizeof msg, "unknown mode '%s'", spec->mode != NULL ? spec->mode : "");
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->dtype == NULL || (strcmp(spec->dtype, "f16") != 0 && strcmp(spec->dtype, "bf16") != 0))
    {
        snprintf(msg, sizeof msg, "unsupported dtype '%s'", spec->dtype != NULL ? spec->dtype : "");
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->vec != 2 && spec->vec != 4 && spec->vec != 8)
    {
        rocke_spec_set_reason(reason, reason_cap, "vec must be 2, 4 or 8");
        return false;
    }
    if(spec->unroll != 1 && spec->unroll != 2 && spec->unroll != 4 && spec->unroll != 8
       && spec->unroll != 16)
    {
        rocke_spec_set_reason(reason, reason_cap, "unroll must be 1/2/4/8/16");
        return false;
    }
    if(!nhwc_is_pow2(spec->tx) || spec->block_size % spec->tx != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "tx must be a power of two dividing block_size");
        return false;
    }
    ty = spec->block_size / spec->tx;
    if(ty & (ty - 1))
    {
        rocke_spec_set_reason(reason, reason_cap, "block_size / tx must be a power of two");
        return false;
    }
    if(!(spec->block_size > 0 && spec->block_size <= rocke_archtarget_max_threads_per_block(target))
       || spec->block_size % spec->wave_size != 0)
    {
        rocke_spec_set_reason(
            reason, reason_cap, "block_size must be a wave multiple within the device limit");
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, (long)2 * spec->block_size * spec->vec * 4))
    {
        rocke_spec_set_reason(reason, reason_cap, "LDS budget exceeded");
        return false;
    }
    return true;
}

bool rocke_batchnorm2d_nhwc_admits(const rocke_batchnorm2d_nhwc_spec_t* spec,
                                   int n,
                                   int c,
                                   int h,
                                   int w,
                                   char* reason,
                                   size_t reason_cap)
{
    long long rows;

    if(reason != NULL && reason_cap > 0)
    {
        reason[0] = '\0';
    }
    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    rows = (long long)n * h * w;
    if(rows <= 0 || c <= 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "empty tensor");
        return false;
    }
    if(c % spec->vec != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "C not a multiple of vec");
        return false;
    }
    if(rocke_batchnorm2d_nhwc_tile_channels(spec) / 2 >= c)
    {
        rocke_spec_set_reason(reason, reason_cap, "channel tile more than twice C");
        return false;
    }
    if(rows * c + (long long)spec->block_size * spec->vec * spec->unroll >= (1LL << 31))
    {
        rocke_spec_set_reason(reason, reason_cap, "element count out of i32 range");
        return false;
    }
    return true;
}

rocke_status_t rocke_batchnorm2d_nhwc_grid(int splits, int tiles, int out[3])
{
    if(out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = splits;
    out[1] = tiles;
    out[2] = 1;
    return ROCKE_OK;
}

long rocke_batchnorm2d_nhwc_workspace_floats(int c, int splits)
{
    return 2L * c * splits;
}

/* ------------------------------------------------------------------ *
 * _NhwcCtx: per-thread tile coordinates shared by every NHWC kernel
 * ------------------------------------------------------------------ */

typedef struct nhwc_ctx
{
    const rocke_batchnorm2d_nhwc_spec_t* spec;
    int VEC, TY, U, CT;
    rocke_value_t* p_c;
    rocke_value_t* tid;
    rocke_value_t* tx;
    rocke_value_t* ty;
    rocke_value_t* split;
    rocke_value_t* ch0;
    rocke_value_t* ch_ok;
    rocke_value_t* ch0s;
    rocke_value_t* col;
    rocke_value_t* rows;
} nhwc_ctx_t;

static void nhwc_ctx_init(rocke_ir_builder_t* b,
                          nhwc_ctx_t* ctx,
                          const rocke_batchnorm2d_nhwc_spec_t* spec,
                          rocke_value_t* p_n,
                          rocke_value_t* p_c,
                          rocke_value_t* p_h,
                          rocke_value_t* p_w,
                          rocke_value_t* split)
{
    rocke_value_t *tile, *t0, *t1;
    ctx->spec = spec;
    ctx->VEC = spec->vec;
    ctx->TY = spec->block_size / spec->tx;
    ctx->U = spec->unroll;
    ctx->CT = spec->tx * spec->vec;
    ctx->p_c = p_c;
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->tx = rocke_b_mod(b, ctx->tid, rocke_b_const_i32(b, spec->tx));
    ctx->ty = rocke_b_div(b, ctx->tid, rocke_b_const_i32(b, spec->tx));
    ctx->split = split;
    tile = rocke_b_block_id_y(b);
    t0 = rocke_b_mul(b, tile, rocke_b_const_i32(b, ctx->CT));
    t1 = rocke_b_mul(b, ctx->tx, rocke_b_const_i32(b, ctx->VEC));
    ctx->ch0 = rocke_b_add(b, t0, t1);
    ctx->ch_ok = rocke_b_cmp_lt(b, ctx->ch0, p_c);
    ctx->ch0s = rocke_b_select(b, ctx->ch_ok, ctx->ch0, rocke_b_const_i32(b, 0));
    ctx->col = rocke_b_mul(b, ctx->tx, rocke_b_const_i32(b, ctx->VEC));
    ctx->rows = rocke_b_mul(b, rocke_b_mul(b, p_n, p_h), p_w);
}

static void nhwc_rows_of_split(rocke_ir_builder_t* b,
                               const nhwc_ctx_t* ctx,
                               rocke_value_t* rows_per_cta,
                               rocke_value_t** r_lo,
                               rocke_value_t** r_hi)
{
    *r_lo = rocke_b_mul(b, ctx->split, rows_per_cta);
    *r_hi = rocke_b_smin(b, ctx->rows, rocke_b_add(b, *r_lo, rows_per_cta));
}

static rocke_value_t* nhwc_owner(rocke_ir_builder_t* b, const nhwc_ctx_t* ctx)
{
    rocke_value_t* e0 = rocke_b_cmp_eq(b, ctx->split, rocke_b_const_i32(b, 0));
    rocke_value_t* e1 = rocke_b_cmp_eq(b, ctx->ty, rocke_b_const_i32(b, 0));
    rocke_value_t* o = rocke_b_land(b, e0, e1);
    return rocke_b_land(b, o, ctx->ch_ok);
}

static rocke_value_t* nhwc_row_index(rocke_ir_builder_t* b,
                                     const nhwc_ctx_t* ctx,
                                     rocke_value_t* r,
                                     rocke_value_t* r_last)
{
    rocke_value_t* rs = rocke_b_smin(b, r, r_last);
    return rocke_b_add(b, rocke_b_mul(b, rs, ctx->p_c), ctx->ch0s);
}

/* max(r_hi - 1, 0) */
static rocke_value_t* nhwc_r_last(rocke_ir_builder_t* b, rocke_value_t* r_hi)
{
    rocke_value_t* t = rocke_b_sub(b, r_hi, rocke_b_const_i32(b, 1));
    return rocke_b_smax(b, t, rocke_b_const_i32(b, 0));
}

/* One loaded row: x (and dy for backward) as vec f32 scalars. */
typedef struct nhwc_loaded
{
    rocke_value_t* xs[NHWC_MAX_VEC];
    rocke_value_t* dys[NHWC_MAX_VEC];
} nhwc_loaded_t;

typedef enum nhwc_op
{
    NHWC_OP_TRAIN = 0, /* train statistics / infer + train streaming output */
    NHWC_OP_BWD = 1
} nhwc_op_t;

/* Kernel-specific state the row loader / accumulator / compute / store read. */
typedef struct nhwc_kern
{
    nhwc_op_t op;
    const char* dtype;
    rocke_value_t* X;
    rocke_value_t* dY;
    rocke_value_t* Out; /* Y or dX */
    rocke_value_t* const* shift; /* train accumulate */
    rocke_value_t* const* mean; /* bwd accumulate / compute */
    rocke_value_t* const* invstd; /* bwd accumulate */
    rocke_value_t* const* sc; /* streaming y = x * sc + sh */
    rocke_value_t* const* sh;
    rocke_value_t* const (*coefs)[3]; /* bwd compute */
} nhwc_kern_t;

static void nhwc_load_rows(rocke_ir_builder_t* b,
                           const nhwc_ctx_t* ctx,
                           const nhwc_kern_t* k,
                           rocke_value_t* idx,
                           nhwc_loaded_t* out)
{
    rocke_b_load_vec_as_f32(b, k->X, idx, k->dtype, ctx->VEC, out->xs);
    if(k->op == NHWC_OP_BWD)
    {
        rocke_b_load_vec_as_f32(b, k->dY, idx, k->dtype, ctx->VEC, out->dys);
    }
}

static void nhwc_accumulate_one(rocke_ir_builder_t* b,
                                const nhwc_ctx_t* ctx,
                                const nhwc_kern_t* k,
                                rocke_value_t** acc,
                                const nhwc_loaded_t* ld,
                                rocke_value_t* valid)
{
    int VEC = ctx->VEC;
    int v;
    for(v = 0; v < VEC; ++v)
    {
        if(k->op == NHWC_OP_TRAIN)
        {
            rocke_value_t* diff = rocke_b_fsub(b, ld->xs[v], k->shift[v]);
            rocke_value_t* d = rocke_b_select(b, valid, diff, rocke_b_const_f32(b, 0.0));
            acc[v] = rocke_b_fadd(b, acc[v], d);
            acc[VEC + v] = rocke_b_fma(b, d, d, acc[VEC + v]);
        }
        else
        {
            rocke_value_t* dym = rocke_b_select(b, valid, ld->dys[v], rocke_b_const_f32(b, 0.0));
            rocke_value_t* xh
                = rocke_b_fmul(b, rocke_b_fsub(b, ld->xs[v], k->mean[v]), k->invstd[v]);
            acc[v] = rocke_b_fadd(b, acc[v], dym);
            acc[VEC + v] = rocke_b_fma(b, dym, xh, acc[VEC + v]);
        }
    }
}

/* _nhwc_accumulate: per-thread accumulation over rows r_lo + ty, + TY, ... */
static void nhwc_accumulate(rocke_ir_builder_t* b,
                            const nhwc_ctx_t* ctx,
                            const nhwc_kern_t* k,
                            rocke_value_t* r_lo,
                            rocke_value_t* r_hi,
                            rocke_value_t** out /* 2 * VEC */)
{
    int n_acc = 2 * ctx->VEC;
    rocke_iter_arg_t inits[2 * NHWC_MAX_VEC];
    char names[2 * NHWC_MAX_VEC][16];
    rocke_value_t *r_last, *lo, *step;
    rocke_for_t for_op;
    int i, u;

    r_last = nhwc_r_last(b, r_hi);
    for(i = 0; i < n_acc; ++i)
    {
        snprintf(names[i], sizeof names[i], "nacc%d", i);
        inits[i].name = names[i];
        inits[i].init = rocke_b_const_f32(b, 0.0);
    }
    lo = rocke_b_add(b, r_lo, ctx->ty);
    step = rocke_b_const_i32(b, ctx->TY * ctx->U);
    for_op = rocke_b_scf_for_iter(b, lo, r_hi, step, inits, n_acc, "nr", false, true);
    rocke_b_region_enter(b, for_op.body);
    {
        nhwc_loaded_t loaded[NHWC_MAX_UNROLL];
        rocke_value_t* valids[NHWC_MAX_UNROLL];
        rocke_value_t* acc[2 * NHWC_MAX_VEC];
        for(u = 0; u < ctx->U; ++u)
        {
            rocke_value_t* ru = rocke_b_add(b, for_op.iv, rocke_b_const_i32(b, u * ctx->TY));
            valids[u] = rocke_b_cmp_lt(b, ru, r_hi);
            nhwc_load_rows(b, ctx, k, nhwc_row_index(b, ctx, ru, r_last), &loaded[u]);
        }
        for(i = 0; i < n_acc; ++i)
        {
            acc[i] = for_op.iter_vars[i];
        }
        for(u = 0; u < ctx->U; ++u)
        {
            nhwc_accumulate_one(b, ctx, k, acc, &loaded[u], valids[u]);
        }
        rocke_b_scf_yield(b, acc, n_acc);
    }
    rocke_b_region_leave(b);
    for(i = 0; i < n_acc; ++i)
    {
        out[i] = for_op.op->results[i];
    }
}

static rocke_value_t* nhwc_lds_load1(rocke_ir_builder_t* b, rocke_value_t* lds, rocke_value_t* slot)
{
    rocke_value_t* idx[1] = {slot};
    return rocke_b_vec_extract(b, rocke_b_smem_load_vN_f32(b, lds, idx, 1, 1), 0);
}

static void nhwc_lds_store1(rocke_ir_builder_t* b,
                            rocke_value_t* lds,
                            rocke_value_t* slot,
                            rocke_value_t* val)
{
    rocke_value_t* idx[1] = {slot};
    rocke_b_smem_store_vN_f32(b, lds, idx, 1, val, 1);
}

/* _nhwc_tile_totals: sum the two per-thread vec-vectors over ty in LDS. */
static void nhwc_tile_totals(rocke_ir_builder_t* b,
                             const nhwc_ctx_t* ctx,
                             rocke_value_t* const* acc,
                             rocke_value_t* lds_a,
                             rocke_value_t* lds_b,
                             rocke_value_t** ta,
                             rocke_value_t** tb)
{
    int VEC = ctx->VEC;
    int CT = ctx->CT;
    rocke_value_t *mine, *slot;
    int v, s;

    mine = rocke_b_add(b, rocke_b_mul(b, ctx->ty, rocke_b_const_i32(b, CT)), ctx->col);
    for(v = 0; v < VEC; ++v)
    {
        slot = rocke_b_add(b, mine, rocke_b_const_i32(b, v));
        nhwc_lds_store1(b, lds_a, slot, acc[v]);
        nhwc_lds_store1(b, lds_b, slot, acc[VEC + v]);
    }
    rocke_b_sync(b);
    for(s = ctx->TY / 2; s >= 1; s /= 2)
    {
        rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_lt(b, ctx->ty, rocke_b_const_i32(b, s)));
        rocke_b_region_enter(b, iff.then_region);
        for(v = 0; v < VEC; ++v)
        {
            rocke_value_t* lds_pair[2] = {lds_a, lds_b};
            rocke_value_t* other;
            int l;
            slot = rocke_b_add(b, mine, rocke_b_const_i32(b, v));
            other = rocke_b_add(b, slot, rocke_b_const_i32(b, s * CT));
            for(l = 0; l < 2; ++l)
            {
                rocke_value_t* x0 = nhwc_lds_load1(b, lds_pair[l], slot);
                rocke_value_t* x1 = nhwc_lds_load1(b, lds_pair[l], other);
                nhwc_lds_store1(b, lds_pair[l], slot, rocke_b_fadd(b, x0, x1));
            }
        }
        rocke_b_region_leave(b);
        rocke_b_sync(b);
    }
    for(v = 0; v < VEC; ++v)
    {
        slot = rocke_b_add(b, ctx->col, rocke_b_const_i32(b, v));
        ta[v] = nhwc_lds_load1(b, lds_a, slot);
        tb[v] = nhwc_lds_load1(b, lds_b, slot);
    }
}

static void nhwc_lds(rocke_ir_builder_t* b,
                     const rocke_batchnorm2d_nhwc_spec_t* spec,
                     rocke_value_t** lds_a,
                     rocke_value_t** lds_b)
{
    rocke_tensor_view_t va, vb;
    int shape[1] = {spec->block_size * spec->vec};
    rocke_make_lds_view(b, &va, rocke_f32(), shape, 1, "lds_a", NULL);
    rocke_make_lds_view(b, &vb, rocke_f32(), shape, 1, "lds_b", NULL);
    *lds_a = va.base;
    *lds_b = vb.base;
}

/* workspace index ((split * 2 + k) * C) + ch */
static rocke_value_t* nhwc_ws_index(
    rocke_ir_builder_t* b, rocke_value_t* split, int k, rocke_value_t* p_c, rocke_value_t* ch)
{
    rocke_value_t* t = rocke_b_mul(b, split, rocke_b_const_i32(b, 2));
    t = rocke_b_add(b, t, rocke_b_const_i32(b, k));
    return rocke_b_add(b, rocke_b_mul(b, t, p_c), ch);
}

/* _nhwc_store_partials: thread row 0 of every channel group stores its two
 * vec-vectors (t = t1 ++ t2). */
static void nhwc_store_partials(rocke_ir_builder_t* b,
                                const nhwc_ctx_t* ctx,
                                rocke_value_t* Workspace,
                                rocke_value_t* const* t,
                                rocke_value_t* p_c)
{
    int VEC = ctx->VEC;
    rocke_value_t* e0 = rocke_b_cmp_eq(b, ctx->ty, rocke_b_const_i32(b, 0));
    rocke_if_t iff = rocke_b_scf_if(b, rocke_b_land(b, e0, ctx->ch_ok));
    int k;
    rocke_b_region_enter(b, iff.then_region);
    for(k = 0; k < 2; ++k)
    {
        rocke_value_t* idx = nhwc_ws_index(b, ctx->split, k, p_c, ctx->ch0);
        rocke_value_t* packed = rocke_b_vec_pack(b, t + k * VEC, VEC, rocke_f32());
        rocke_b_global_store_vN(b, Workspace, idx, packed, VEC, 0);
    }
    rocke_b_region_leave(b);
}

/* _nhwc_merge_partials: per-thread sum of the planar partials s = ty, ty + TY, ... */
static void nhwc_merge_partials(rocke_ir_builder_t* b,
                                const nhwc_ctx_t* ctx,
                                rocke_value_t* Workspace,
                                rocke_value_t* p_c,
                                rocke_value_t* p_splits,
                                rocke_value_t** out)
{
    int VEC = ctx->VEC;
    int n_acc = 2 * VEC;
    rocke_iter_arg_t inits[2 * NHWC_MAX_VEC];
    char names[2 * NHWC_MAX_VEC][16];
    rocke_value_t* step;
    rocke_for_t for_op;
    int i, k, v;

    for(i = 0; i < n_acc; ++i)
    {
        snprintf(names[i], sizeof names[i], "mrg%d", i);
        inits[i].name = names[i];
        inits[i].init = rocke_b_const_f32(b, 0.0);
    }
    step = rocke_b_const_i32(b, ctx->TY);
    for_op = rocke_b_scf_for_iter(b, ctx->ty, p_splits, step, inits, n_acc, "ms", false, true);
    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* acc[2 * NHWC_MAX_VEC];
        for(k = 0; k < 2; ++k)
        {
            rocke_value_t* idx = nhwc_ws_index(b, for_op.iv, k, p_c, ctx->ch0s);
            /* load_vec(b, Workspace, idx, dtype="f32", n=VEC) is
             * b.global_load_vN(Workspace, idx, F32, VEC). */
            rocke_value_t* vals = rocke_b_global_load_vN(b, Workspace, idx, rocke_f32(), VEC, 0);
            for(v = 0; v < VEC; ++v)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, vals, v);
                acc[k * VEC + v] = rocke_b_fadd(b, for_op.iter_vars[k * VEC + v], e);
            }
        }
        rocke_b_scf_yield(b, acc, n_acc);
    }
    rocke_b_region_leave(b);
    for(i = 0; i < n_acc; ++i)
    {
        out[i] = for_op.op->results[i];
    }
}

static void nhwc_compute(rocke_ir_builder_t* b,
                         const nhwc_ctx_t* ctx,
                         const nhwc_kern_t* k,
                         const nhwc_loaded_t* ld,
                         rocke_value_t** ys)
{
    int v;
    for(v = 0; v < ctx->VEC; ++v)
    {
        if(k->op == NHWC_OP_TRAIN)
        {
            ys[v] = rocke_b_fma(b, ld->xs[v], k->sc[v], k->sh[v]);
        }
        else
        {
            rocke_value_t* t_a = rocke_b_fmul(b, k->coefs[v][0], ld->dys[v]);
            rocke_value_t* d = rocke_b_fsub(b, ld->xs[v], k->mean[v]);
            rocke_value_t* t_k2 = rocke_b_fmul(b, k->coefs[v][1], d);
            ys[v] = rocke_b_fsub(b, rocke_b_fsub(b, t_a, t_k2), k->coefs[v][2]);
        }
    }
}

/* _nhwc_apply_rows: streaming pass over this thread's rows. */
static void nhwc_apply_rows(rocke_ir_builder_t* b,
                            const nhwc_ctx_t* ctx,
                            const nhwc_kern_t* k,
                            rocke_value_t* r_lo,
                            rocke_value_t* r_hi)
{
    rocke_value_t *r_last, *lo, *step;
    rocke_for_t for_op;
    int u;

    r_last = nhwc_r_last(b, r_hi);
    lo = rocke_b_add(b, r_lo, ctx->ty);
    step = rocke_b_const_i32(b, ctx->TY * ctx->U);
    for_op = rocke_b_scf_for(b, lo, r_hi, step, "ar");
    rocke_b_region_enter(b, for_op.body);
    {
        nhwc_loaded_t loaded[NHWC_MAX_UNROLL];
        rocke_value_t* valids[NHWC_MAX_UNROLL];
        rocke_value_t* idxs[NHWC_MAX_UNROLL];
        for(u = 0; u < ctx->U; ++u)
        {
            rocke_value_t* ru = rocke_b_add(b, for_op.iv, rocke_b_const_i32(b, u * ctx->TY));
            rocke_value_t* lt = rocke_b_cmp_lt(b, ru, r_hi);
            valids[u] = rocke_b_land(b, lt, ctx->ch_ok);
            idxs[u] = nhwc_row_index(b, ctx, ru, r_last);
            nhwc_load_rows(b, ctx, k, idxs[u], &loaded[u]);
        }
        for(u = 0; u < ctx->U; ++u)
        {
            rocke_value_t* ys[NHWC_MAX_VEC];
            rocke_if_t iff;
            nhwc_compute(b, ctx, k, &loaded[u], ys);
            iff = rocke_b_scf_if(b, valids[u]);
            rocke_b_region_enter(b, iff.then_region);
            rocke_b_store_vec(
                b, k->Out, idxs[u], rocke_b_pack_f32_to(b, ys, ctx->VEC, k->dtype), ctx->VEC);
            rocke_b_region_leave(b);
        }
    }
    rocke_b_region_leave(b);
}

/* _nhwc_f32_scalars: [load_scalar_as_f32(ptr, ch0s + v) for v]. */
static void nhwc_f32_scalars(rocke_ir_builder_t* b,
                             const nhwc_ctx_t* ctx,
                             rocke_value_t* ptr,
                             const char* dtype,
                             rocke_value_t** out)
{
    int v;
    for(v = 0; v < ctx->VEC; ++v)
    {
        rocke_value_t* idx = rocke_b_add(b, ctx->ch0s, rocke_b_const_i32(b, v));
        out[v] = rocke_b_load_scalar_as_f32(b, ptr, idx, dtype);
    }
}

/* [global_load_f32(ptr, ch0s + v) for v]. */
static void nhwc_f32_vector(rocke_ir_builder_t* b,
                            const nhwc_ctx_t* ctx,
                            rocke_value_t* ptr,
                            rocke_value_t** out)
{
    int v;
    for(v = 0; v < ctx->VEC; ++v)
    {
        rocke_value_t* idx = rocke_b_add(b, ctx->ch0s, rocke_b_const_i32(b, v));
        out[v] = rocke_b_global_load_f32(b, ptr, idx, 4);
    }
}

/* _nhwc_scale_shift: per-channel (scale, shift) with y = x * scale + shift. */
static void nhwc_scale_shift(rocke_ir_builder_t* b,
                             const nhwc_ctx_t* ctx,
                             const char* dtype,
                             rocke_value_t* Gamma,
                             rocke_value_t* Beta,
                             rocke_value_t* const* mean,
                             rocke_value_t* const* var,
                             rocke_value_t* eps,
                             rocke_value_t** sc,
                             rocke_value_t** sh)
{
    rocke_value_t* g[NHWC_MAX_VEC];
    rocke_value_t* bt[NHWC_MAX_VEC];
    int v;
    nhwc_f32_scalars(b, ctx, Gamma, dtype, g);
    nhwc_f32_scalars(b, ctx, Beta, dtype, bt);
    for(v = 0; v < ctx->VEC; ++v)
    {
        rocke_value_t* r = rocke_b_rsqrt(b, rocke_b_fadd(b, var[v], eps));
        sc[v] = rocke_b_fmul(b, r, g[v]);
    }
    for(v = 0; v < ctx->VEC; ++v)
    {
        sh[v] = rocke_b_fsub(b, bt[v], rocke_b_fmul(b, mean[v], sc[v]));
    }
}

static bool nhwc_check(rocke_ir_builder_t* b,
                       const rocke_batchnorm2d_nhwc_spec_t* spec,
                       const char* mode,
                       const char* kind,
                       const char* what)
{
    char why[256];
    bool ok = rocke_batchnorm2d_is_valid_nhwc_spec(spec, NULL, why, sizeof why);
    bool kind_ok = true;
    if(strcmp(mode, "infer") != 0)
    {
        kind_ok = kind != NULL
                  && (strcmp(kind, "fused") == 0 || strcmp(kind, "reduce") == 0
                      || strcmp(kind, "apply") == 0);
    }
    if(!ok || strcmp(spec->mode, mode) != 0 || !kind_ok)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "invalid NHWC %s spec: %s",
                        what,
                        why[0] != '\0' ? why : spec->mode);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ *
 * infer
 * ------------------------------------------------------------------ */

rocke_kernel_def_t* rocke_build_batchnorm2d_nhwc_infer(rocke_ir_builder_t* b,
                                                       const rocke_batchnorm2d_nhwc_spec_t* spec)
{
    const rocke_type_t* io_ty;
    rocke_value_t *X, *Gamma, *Beta, *Y, *RunningMean, *RunningVar;
    rocke_value_t *p_n, *p_c, *p_h, *p_w, *eps, *p_rpc, *split, *r_lo, *r_hi;
    rocke_value_t* mean[NHWC_MAX_VEC];
    rocke_value_t* var[NHWC_MAX_VEC];
    rocke_value_t* sc[NHWC_MAX_VEC];
    rocke_value_t* sh[NHWC_MAX_VEC];
    nhwc_ctx_t ctx;
    nhwc_kern_t k;

    if(b == NULL || spec == NULL || !nhwc_check(b, spec, "infer", NULL, "infer"))
    {
        return NULL;
    }
    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    X = rocke_bn_ptr_param(b, "X", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    Gamma = rocke_bn_ptr_param(b, "Gamma", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    Beta = rocke_bn_ptr_param(b, "Beta", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    Y = rocke_bn_ptr_param(b, "Y", io_ty, ROCKE_BN_PARAM_WRITEONLY, 16);
    RunningMean = rocke_bn_ptr_param(b, "RunningMean", rocke_f32(), ROCKE_BN_PARAM_PLAIN, 0);
    RunningVar = rocke_bn_ptr_param(b, "RunningVar", rocke_f32(), ROCKE_BN_PARAM_PLAIN, 0);
    p_n = rocke_b_param(b, "N", rocke_i32(), NULL);
    p_c = rocke_b_param(b, "C", rocke_i32(), NULL);
    p_h = rocke_b_param(b, "H", rocke_i32(), NULL);
    p_w = rocke_b_param(b, "W", rocke_i32(), NULL);
    eps = rocke_b_param(b, "eps", rocke_f32(), NULL);
    (void)rocke_b_param(b, "momentum", rocke_f32(), NULL);
    p_rpc = rocke_b_param(b, "RowsPerCta", rocke_i32(), NULL);

    split = rocke_b_block_id_x(b);
    nhwc_ctx_init(b, &ctx, spec, p_n, p_c, p_h, p_w, split);
    nhwc_f32_vector(b, &ctx, RunningMean, mean);
    nhwc_f32_vector(b, &ctx, RunningVar, var);
    nhwc_scale_shift(b, &ctx, spec->dtype, Gamma, Beta, mean, var, eps, sc, sh);
    nhwc_rows_of_split(b, &ctx, p_rpc, &r_lo, &r_hi);

    memset(&k, 0, sizeof k);
    k.op = NHWC_OP_TRAIN;
    k.dtype = spec->dtype;
    k.X = X;
    k.Out = Y;
    k.sc = sc;
    k.sh = sh;
    nhwc_apply_rows(b, &ctx, &k, r_lo, r_hi);

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

/* ------------------------------------------------------------------ *
 * training forward
 * ------------------------------------------------------------------ */

typedef struct nhwc_params
{
    rocke_value_t *X, *dY, *Workspace, *Gamma, *Beta, *Y, *RunningMean, *RunningVar, *Mean, *InvStd;
    rocke_value_t *Scale, *dX, *dScale, *dBias;
    rocke_value_t *N, *C, *H, *W, *eps, *momentum, *RowsPerCta, *Splits;
} nhwc_params_t;

static void nhwc_dims_params(rocke_ir_builder_t* b, nhwc_params_t* P)
{
    P->N = rocke_b_param(b, "N", rocke_i32(), NULL);
    P->C = rocke_b_param(b, "C", rocke_i32(), NULL);
    P->H = rocke_b_param(b, "H", rocke_i32(), NULL);
    P->W = rocke_b_param(b, "W", rocke_i32(), NULL);
}

/* _nhwc_train_params */
static void nhwc_train_params(rocke_ir_builder_t* b,
                              const rocke_type_t* io_ty,
                              const char* kind,
                              nhwc_params_t* P)
{
    bool reduce = strcmp(kind, "reduce") == 0;
    bool apply = strcmp(kind, "apply") == 0;
    bool fused = strcmp(kind, "fused") == 0;

    memset(P, 0, sizeof *P);
    P->X = rocke_bn_ptr_param(b, "X", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    if(reduce)
    {
        P->Workspace = rocke_bn_ptr_param(b, "Workspace", rocke_f32(), ROCKE_BN_PARAM_PLAIN, 0);
    }
    else
    {
        if(apply)
        {
            P->Workspace
                = rocke_bn_ptr_param(b, "Workspace", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
        }
        P->Gamma = rocke_bn_ptr_param(b, "Gamma", io_ty, ROCKE_BN_PARAM_READONLY, 16);
        P->Beta = rocke_bn_ptr_param(b, "Beta", io_ty, ROCKE_BN_PARAM_READONLY, 16);
        P->Y = rocke_bn_ptr_param(b, "Y", io_ty, ROCKE_BN_PARAM_WRITEONLY, 16);
        P->RunningMean = rocke_bn_ptr_param(b, "RunningMean", rocke_f32(), ROCKE_BN_PARAM_PLAIN, 0);
        P->RunningVar = rocke_bn_ptr_param(b, "RunningVar", rocke_f32(), ROCKE_BN_PARAM_PLAIN, 0);
        P->Mean = rocke_bn_ptr_param(b, "Mean", io_ty, ROCKE_BN_PARAM_WRITEONLY, 0);
        P->InvStd = rocke_bn_ptr_param(b, "InvStd", io_ty, ROCKE_BN_PARAM_WRITEONLY, 0);
    }
    nhwc_dims_params(b, P);
    if(!reduce)
    {
        P->eps = rocke_b_param(b, "eps", rocke_f32(), NULL);
        P->momentum = rocke_b_param(b, "momentum", rocke_f32(), NULL);
    }
    if(!fused)
    {
        P->RowsPerCta = rocke_b_param(b, "RowsPerCta", rocke_i32(), NULL);
    }
    if(apply)
    {
        P->Splits = rocke_b_param(b, "Splits", rocke_i32(), NULL);
    }
}

rocke_kernel_def_t* rocke_build_batchnorm2d_nhwc_train(rocke_ir_builder_t* b,
                                                       const rocke_batchnorm2d_nhwc_spec_t* spec,
                                                       const char* kind)
{
    const rocke_type_t* io_ty;
    nhwc_params_t P;
    nhwc_ctx_t ctx;
    nhwc_kern_t k;
    rocke_value_t *split, *lds_a, *lds_b, *r_lo, *r_hi, *m_f, *inv_m;
    rocke_value_t* shift[NHWC_MAX_VEC];
    rocke_value_t* acc[2 * NHWC_MAX_VEC];
    rocke_value_t* t[2 * NHWC_MAX_VEC];
    rocke_value_t* mean[NHWC_MAX_VEC];
    rocke_value_t* var[NHWC_MAX_VEC];
    rocke_value_t* inv[NHWC_MAX_VEC];
    rocke_value_t* sc[NHWC_MAX_VEC];
    rocke_value_t* sh[NHWC_MAX_VEC];
    int VEC, v;
    bool fused;

    if(b == NULL || spec == NULL || !nhwc_check(b, spec, "train", kind, "training"))
    {
        return NULL;
    }
    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    VEC = spec->vec;
    fused = strcmp(kind, "fused") == 0;
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    nhwc_train_params(b, io_ty, kind, &P);
    split = fused ? rocke_b_const_i32(b, 0) : rocke_b_block_id_x(b);
    nhwc_ctx_init(b, &ctx, spec, P.N, P.C, P.H, P.W, split);
    nhwc_lds(b, spec, &lds_a, &lds_b);
    /* _nhwc_train_shift: channel shift = row 0 of the channel. */
    rocke_b_load_vec_as_f32(b, P.X, ctx.ch0s, spec->dtype, VEC, shift);

    memset(&k, 0, sizeof k);
    k.op = NHWC_OP_TRAIN;
    k.dtype = spec->dtype;
    k.X = P.X;
    k.shift = shift;
    if(strcmp(kind, "apply") == 0)
    {
        nhwc_merge_partials(b, &ctx, P.Workspace, P.C, P.Splits, acc);
    }
    else
    {
        if(fused)
        {
            r_lo = rocke_b_const_i32(b, 0);
            r_hi = ctx.rows;
        }
        else
        {
            nhwc_rows_of_split(b, &ctx, P.RowsPerCta, &r_lo, &r_hi);
        }
        nhwc_accumulate(b, &ctx, &k, r_lo, r_hi, acc);
    }
    nhwc_tile_totals(b, &ctx, acc, lds_a, lds_b, t, t + VEC);
    if(strcmp(kind, "reduce") == 0)
    {
        nhwc_store_partials(b, &ctx, P.Workspace, t, P.C);
        return rocke_ir_builder_ok(b) ? b->kernel : NULL;
    }

    /* _nhwc_train_stats */
    m_f = rocke_b_sitofp_f32(b, ctx.rows);
    inv_m = rocke_b_fdiv(b, rocke_b_const_f32(b, 1.0), m_f);
    for(v = 0; v < VEC; ++v)
    {
        rocke_value_t* ms = rocke_b_fmul(b, t[v], inv_m);
        rocke_value_t* e2 = rocke_b_fmul(b, t[VEC + v], inv_m);
        rocke_value_t* m2 = rocke_b_fmul(b, ms, ms);
        rocke_value_t* d = rocke_b_fsub(b, e2, m2);
        rocke_value_t* vr = rocke_b_fmax(b, d, rocke_b_const_f32(b, 0.0));
        mean[v] = rocke_b_fadd(b, shift[v], ms);
        var[v] = vr;
        inv[v] = rocke_b_rsqrt(b, rocke_b_fadd(b, vr, P.eps));
    }

    /* _nhwc_train_epilogue */
    {
        rocke_if_t iff = rocke_b_scf_if(b, nhwc_owner(b, &ctx));
        rocke_value_t *one_minus, *unbias, *den;
        rocke_b_region_enter(b, iff.then_region);
        one_minus = rocke_b_fsub(b, rocke_b_const_f32(b, 1.0), P.momentum);
        den = rocke_b_fsub(b, m_f, rocke_b_const_f32(b, 1.0));
        den = rocke_b_fmax(b, den, rocke_b_const_f32(b, 1.0));
        unbias = rocke_b_fdiv(b, m_f, den);
        for(v = 0; v < VEC; ++v)
        {
            rocke_value_t* ch = rocke_b_add(b, ctx.ch0, rocke_b_const_i32(b, v));
            rocke_value_t* old_m = rocke_b_global_load_f32(b, P.RunningMean, ch, 4);
            rocke_value_t* old_v = rocke_b_global_load_f32(b, P.RunningVar, ch, 4);
            rocke_value_t* a0 = rocke_b_fmul(b, one_minus, old_m);
            rocke_value_t* a1 = rocke_b_fmul(b, P.momentum, mean[v]);
            rocke_b_global_store(b, P.RunningMean, ch, rocke_b_fadd(b, a0, a1), 1);
            a0 = rocke_b_fmul(b, one_minus, old_v);
            a1 = rocke_b_fmul(b, P.momentum, rocke_b_fmul(b, var[v], unbias));
            rocke_b_global_store(b, P.RunningVar, ch, rocke_b_fadd(b, a0, a1), 1);
            rocke_b_store_scalar_from_f32(b, P.Mean, ch, mean[v], spec->dtype);
            rocke_b_store_scalar_from_f32(b, P.InvStd, ch, inv[v], spec->dtype);
        }
        rocke_b_region_leave(b);
    }

    nhwc_scale_shift(b, &ctx, spec->dtype, P.Gamma, P.Beta, mean, var, P.eps, sc, sh);
    k.Out = P.Y;
    k.sc = sc;
    k.sh = sh;
    if(fused)
    {
        r_lo = rocke_b_const_i32(b, 0);
        r_hi = ctx.rows;
    }
    else
    {
        nhwc_rows_of_split(b, &ctx, P.RowsPerCta, &r_lo, &r_hi);
    }
    nhwc_apply_rows(b, &ctx, &k, r_lo, r_hi);

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

/* ------------------------------------------------------------------ *
 * backward
 * ------------------------------------------------------------------ */

/* _nhwc_bwd_params */
static void nhwc_bwd_params(rocke_ir_builder_t* b,
                            const rocke_type_t* io_ty,
                            const char* kind,
                            nhwc_params_t* P)
{
    bool reduce = strcmp(kind, "reduce") == 0;
    bool apply = strcmp(kind, "apply") == 0;
    bool fused = strcmp(kind, "fused") == 0;

    memset(P, 0, sizeof *P);
    P->X = rocke_bn_ptr_param(b, "X", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    P->dY = rocke_bn_ptr_param(b, "dY", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    if(!reduce)
    {
        P->Scale = rocke_bn_ptr_param(b, "Scale", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    }
    P->Mean = rocke_bn_ptr_param(b, "Mean", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    P->InvStd = rocke_bn_ptr_param(b, "InvStd", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    if(!fused)
    {
        P->Workspace
            = rocke_bn_ptr_param(b,
                                 "Workspace",
                                 rocke_f32(),
                                 apply ? ROCKE_BN_PARAM_READONLY : ROCKE_BN_PARAM_NOT_READONLY,
                                 0);
    }
    if(!reduce)
    {
        P->dX = rocke_bn_ptr_param(b, "dX", io_ty, ROCKE_BN_PARAM_WRITEONLY, 16);
        P->dScale = rocke_bn_ptr_param(b, "dScale", rocke_f32(), ROCKE_BN_PARAM_WRITEONLY, 0);
        P->dBias = rocke_bn_ptr_param(b, "dBias", rocke_f32(), ROCKE_BN_PARAM_WRITEONLY, 0);
    }
    nhwc_dims_params(b, P);
    if(!fused)
    {
        P->RowsPerCta = rocke_b_param(b, "RowsPerCta", rocke_i32(), NULL);
    }
    if(apply)
    {
        P->Splits = rocke_b_param(b, "Splits", rocke_i32(), NULL);
    }
}

rocke_kernel_def_t* rocke_build_batchnorm2d_nhwc_bwd(rocke_ir_builder_t* b,
                                                     const rocke_batchnorm2d_nhwc_spec_t* spec,
                                                     const char* kind)
{
    const rocke_type_t* io_ty;
    nhwc_params_t P;
    nhwc_ctx_t ctx;
    nhwc_kern_t k;
    rocke_value_t *split, *lds_a, *lds_b, *r_lo, *r_hi, *m_f;
    rocke_value_t* mean[NHWC_MAX_VEC];
    rocke_value_t* invstd[NHWC_MAX_VEC];
    rocke_value_t* scale[NHWC_MAX_VEC];
    rocke_value_t* acc[2 * NHWC_MAX_VEC];
    rocke_value_t* t[2 * NHWC_MAX_VEC];
    rocke_value_t* coefs[NHWC_MAX_VEC][3];
    int VEC, v;
    bool fused;

    if(b == NULL || spec == NULL || !nhwc_check(b, spec, "bwd", kind, "backward"))
    {
        return NULL;
    }
    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    VEC = spec->vec;
    fused = strcmp(kind, "fused") == 0;
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    nhwc_bwd_params(b, io_ty, kind, &P);
    split = fused ? rocke_b_const_i32(b, 0) : rocke_b_block_id_x(b);
    nhwc_ctx_init(b, &ctx, spec, P.N, P.C, P.H, P.W, split);
    nhwc_lds(b, spec, &lds_a, &lds_b);
    nhwc_f32_vector(b, &ctx, P.Mean, mean);
    nhwc_f32_vector(b, &ctx, P.InvStd, invstd);
    if(fused)
    {
        r_lo = rocke_b_const_i32(b, 0);
        r_hi = ctx.rows;
    }
    else
    {
        nhwc_rows_of_split(b, &ctx, P.RowsPerCta, &r_lo, &r_hi);
    }

    memset(&k, 0, sizeof k);
    k.op = NHWC_OP_BWD;
    k.dtype = spec->dtype;
    k.X = P.X;
    k.dY = P.dY;
    k.mean = mean;
    k.invstd = invstd;
    if(strcmp(kind, "apply") == 0)
    {
        nhwc_merge_partials(b, &ctx, P.Workspace, P.C, P.Splits, acc);
    }
    else
    {
        nhwc_accumulate(b, &ctx, &k, r_lo, r_hi, acc);
    }
    /* t[0..VEC) = sum(dy), t[VEC..2VEC) = sum(dy * xhat) */
    nhwc_tile_totals(b, &ctx, acc, lds_a, lds_b, t, t + VEC);
    if(strcmp(kind, "reduce") == 0)
    {
        nhwc_store_partials(b, &ctx, P.Workspace, t, P.C);
        return rocke_ir_builder_ok(b) ? b->kernel : NULL;
    }

    {
        rocke_if_t iff = rocke_b_scf_if(b, nhwc_owner(b, &ctx));
        rocke_b_region_enter(b, iff.then_region);
        for(v = 0; v < VEC; ++v)
        {
            rocke_value_t* ch = rocke_b_add(b, ctx.ch0, rocke_b_const_i32(b, v));
            rocke_b_global_store(b, P.dScale, ch, t[VEC + v], 1);
            rocke_b_global_store(b, P.dBias, ch, t[v], 1);
        }
        rocke_b_region_leave(b);
    }
    nhwc_f32_vector(b, &ctx, P.Scale, scale);
    m_f = rocke_b_sitofp_f32(b, ctx.rows);
    for(v = 0; v < VEC; ++v)
    {
        rocke_bn_bwd_coefs(b, t[v], t[VEC + v], scale[v], mean[v], invstd[v], m_f, coefs[v]);
    }
    k.Out = P.dX;
    k.coefs = coefs;
    nhwc_apply_rows(b, &ctx, &k, r_lo, r_hi);

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_batchnorm2d_nhwc(rocke_ir_builder_t* b,
                                                 const rocke_batchnorm2d_nhwc_spec_t* spec,
                                                 const char* kind)
{
    if(spec == NULL || spec->mode == NULL)
    {
        return NULL;
    }
    if(strcmp(spec->mode, "infer") == 0)
    {
        return rocke_build_batchnorm2d_nhwc_infer(b, spec);
    }
    if(strcmp(spec->mode, "train") == 0)
    {
        return rocke_build_batchnorm2d_nhwc_train(b, spec, kind);
    }
    return rocke_build_batchnorm2d_nhwc_bwd(b, spec, kind);
}

rocke_kernel_def_t* rocke_build_batchnorm2d_nhwc_new(rocke_ir_builder_t* b,
                                                     const rocke_batchnorm2d_nhwc_spec_t* spec,
                                                     const char* kind)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        const char* name_kind = kind;
        if(b == NULL || spec == NULL || spec->mode == NULL)
        {
            return NULL;
        }
        if(strcmp(spec->mode, "infer") == 0)
        {
            name_kind = "infer";
        }
        if(rocke_batchnorm2d_nhwc_kernel_name(spec, name_kind, name, sizeof name) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_batchnorm2d_nhwc(b, spec, kind);
    });
}
