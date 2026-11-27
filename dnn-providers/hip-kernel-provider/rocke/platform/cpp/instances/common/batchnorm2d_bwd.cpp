// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * batchnorm2d_bwd.cpp -- C99 port of the dynamic NCHW BatchNorm2D backward
 * kernels of rocke/instances/common/batchnorm2d.py: the single-launch
 * build_batchnorm2d_bwd_dynamic and the split pair
 * build_batchnorm2d_bwd_split_reduce / build_batchnorm2d_bwd_split_apply.
 *
 * Byte-faithful reproduction of the Python builder-call sequence: every
 * builder call that emits IR is issued in Python's left-to-right evaluation
 * order (no two emitting calls share one C argument list).
 */

#include "rocke/instance_batchnorm2d_bwd.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.sweep.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/instance_batchnorm2d.h"
#include "rocke/instance_batchnorm2d_bwd_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ------------------------------------------------------------------ *
 * small shared pieces
 * ------------------------------------------------------------------ */

static void bwd_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

rocke_value_t*
    rocke_bn_fadd_combine(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, void* user)
{
    (void)user;
    return rocke_b_fadd(b, a, c);
}

/* b.param(name, PtrType(t, "global"), noalias=True, [readonly|writeonly]=True, [align=16]). */
rocke_value_t* rocke_bn_ptr_param(
    rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem, int access, int align)
{
    rocke_param_opts_t opts;
    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    if(access == ROCKE_BN_PARAM_READONLY || access == ROCKE_BN_PARAM_NOT_READONLY)
    {
        opts.readonly = (access == ROCKE_BN_PARAM_READONLY);
        opts.readonly_set = true;
    }
    else if(access == ROCKE_BN_PARAM_WRITEONLY)
    {
        opts.writeonly = true;
        opts.writeonly_set = true;
    }
    if(align > 0)
    {
        opts.align = align;
        opts.align_set = true;
    }
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &opts);
}

/* _bwd_coefs(b, s1, s2, scale, mean, invstd, m_f) -> (a, k2, k3). */
void rocke_bn_bwd_coefs(rocke_ir_builder_t* b,
                        rocke_value_t* s1,
                        rocke_value_t* s2,
                        rocke_value_t* scale,
                        rocke_value_t* mean,
                        rocke_value_t* invstd,
                        rocke_value_t* m_f,
                        rocke_value_t* out[3])
{
    rocke_value_t *one, *inv_m, *a, *t0, *t1, *k2, *k3;
    (void)mean;
    one = rocke_b_const_f32(b, 1.0);
    inv_m = rocke_b_fdiv(b, one, m_f);
    a = rocke_b_fmul(b, scale, invstd);
    t0 = rocke_b_fmul(b, a, invstd);
    t1 = rocke_b_fmul(b, s2, inv_m);
    k2 = rocke_b_fmul(b, t0, t1);
    t1 = rocke_b_fmul(b, s1, inv_m);
    k3 = rocke_b_fmul(b, a, t1);
    out[0] = a;
    out[1] = k2;
    out[2] = k3;
}

/* _bwd_block_sums: block-wide sums of two per-thread f32 partials. */
static void bwd_block_sums(rocke_ir_builder_t* b,
                           int block_size,
                           bool fast_reduce,
                           int wave_size,
                           rocke_value_t* s1,
                           rocke_value_t* s2,
                           rocke_value_t* lds_a,
                           rocke_value_t* lds_c,
                           rocke_value_t* tid,
                           rocke_value_t** t1,
                           rocke_value_t** t2)
{
    if(fast_reduce)
    {
        *t1 = rocke_block_lds_reduce_with_wave_prologue(
            b, s1, lds_a, tid, block_size, ROCKE_REDUCE_SUM, wave_size);
        *t2 = rocke_block_lds_reduce_with_wave_prologue(
            b, s2, lds_c, tid, block_size, ROCKE_REDUCE_SUM, wave_size);
        return;
    }
    (void)rocke_block_lds_reduce_pair(
        b, s1, s2, lds_a, lds_c, tid, block_size, ROCKE_REDUCE_SUM, ROCKE_REDUCE_SUM, t1, t2);
}

/* row_of(og): NG == 1 -> og; else og * NG + lidihw; row = n * C + channel. */
typedef struct bwd_rows
{
    int ng;
    rocke_value_t* lidihw;
    rocke_value_t* p_c;
    rocke_value_t* channel;
} bwd_rows_t;

static rocke_value_t* bwd_row_of(rocke_ir_builder_t* b, const bwd_rows_t* r, rocke_value_t* og)
{
    rocke_value_t* n_rt = og;
    if(r->ng != 1)
    {
        rocke_value_t* c = rocke_b_const_i32(b, r->ng);
        n_rt = rocke_b_add(b, rocke_b_mul(b, og, c), r->lidihw);
    }
    return rocke_b_add(b, rocke_b_mul(b, n_rt, r->p_c), r->channel);
}

/* Shared inputs of _bwd_accumulate / _bwd_apply_rows. */
typedef struct bwd_sweep
{
    int vec;
    const char* dtype;
    rocke_value_t *X, *dY, *dX;
    const rocke_tensor_view_t *x_view, *dy_view, *dx_view;
    rocke_value_t* HW;
    rocke_value_t* mean_c;
    rocke_value_t* invstd_c;
    rocke_value_t* tid_row;
    int threads;
    rocke_value_t *n_begin, *n_end;
    const bwd_rows_t* rows;
    rocke_value_t* hw_lo;
    rocke_value_t* length;
    rocke_value_t* seg_last;
    const char* iv_prefix;
} bwd_sweep_t;

typedef struct bwd_acc_ctx
{
    const bwd_sweep_t* s;
    const rocke_tile_window_t* dy_tile;
} bwd_acc_ctx_t;

static void bwd_acc_body(rocke_ir_builder_t* b,
                         rocke_value_t* n_off,
                         rocke_value_t* const* x_scalars,
                         int vec,
                         rocke_value_t* const* iter_vars,
                         int num_iter_vars,
                         rocke_value_t** new_iter_vals,
                         void* user)
{
    bwd_acc_ctx_t* ctx = (bwd_acc_ctx_t*)user;
    rocke_value_t* dys[8];
    rocke_value_t* prods[8];
    rocke_value_t* li[2];
    rocke_value_t* part;
    int i;
    (void)num_iter_vars;

    li[0] = rocke_b_const_i32(b, 0);
    li[1] = n_off;
    rocke_tile_window_load_vec_as_f32(b, ctx->dy_tile, li, 2, vec, dys);
    for(i = 0; i < vec; ++i)
    {
        rocke_value_t* d = rocke_b_fsub(b, x_scalars[i], ctx->s->mean_c);
        rocke_value_t* xh = rocke_b_fmul(b, d, ctx->s->invstd_c);
        prods[i] = rocke_b_fmul(b, dys[i], xh);
    }
    part = rocke_tree_reduce(b, rocke_bn_fadd_combine, NULL, dys, vec);
    new_iter_vals[0] = rocke_b_fadd(b, iter_vars[0], part);
    part = rocke_tree_reduce(b, rocke_bn_fadd_combine, NULL, prods, vec);
    new_iter_vals[1] = rocke_b_fadd(b, iter_vars[1], part);
}

/* _bwd_accumulate: per-thread (sum(dy), sum(dy * xhat)) over rows
 * n_begin .. n_end, each swept over [hw_lo, hw_lo + length). */
static void bwd_accumulate(rocke_ir_builder_t* b, const bwd_sweep_t* s, rocke_value_t* out[2])
{
    rocke_iter_arg_t outer_args[2];
    rocke_for_t for_op;
    rocke_value_t* step;
    char iv_n[64];
    char iv_hw[64];
    int VEC = s->vec;
    int tc;

    snprintf(iv_n, sizeof iv_n, "%s_n", s->iv_prefix);
    snprintf(iv_hw, sizeof iv_hw, "%s_hw", s->iv_prefix);

    step = rocke_b_const_i32(b, 1);
    outer_args[0].name = "s_dy";
    outer_args[0].init = rocke_b_const_f32(b, 0.0);
    outer_args[1].name = "s_dyx";
    outer_args[1].init = rocke_b_const_f32(b, 0.0);
    for_op = rocke_b_scf_for_iter(b, s->n_begin, s->n_end, step, outer_args, 2, iv_n, false, true);
    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* row_n = bwd_row_of(b, s->rows, for_op.iv);
        rocke_tile_window_t x_tile, dy_tile;
        rocke_iter_arg_t inner_args[2];
        rocke_sweep_dynamic_result_t res;
        rocke_value_t *s1, *s2;
        rocke_value_t* new_vals[2];
        bwd_acc_ctx_t actx;
        int lengths2[2] = {1, 1};
        rocke_value_t* origin[2];

        origin[0] = row_n;
        origin[1] = s->hw_lo;
        rocke_make_tile_window(&x_tile, s->x_view, lengths2, origin, 2);
        rocke_make_tile_window(&dy_tile, s->dy_view, lengths2, origin, 2);

        inner_args[0].name = "s_dy_in";
        inner_args[0].init = for_op.iter_vars[0];
        inner_args[1].name = "s_dyx_in";
        inner_args[1].init = for_op.iter_vars[1];
        actx.s = s;
        actx.dy_tile = &dy_tile;
        res = rocke_sweep_row_chunks_dynamic(b,
                                             &x_tile,
                                             s->tid_row,
                                             s->threads,
                                             VEC,
                                             s->length,
                                             inner_args,
                                             2,
                                             NULL,
                                             bwd_acc_body,
                                             &actx,
                                             iv_hw);
        s1 = res.final_iter_vals[0];
        s2 = res.final_iter_vals[1];

        for(tc = 0; tc < VEC; ++tc)
        {
            rocke_value_t* c_off = rocke_b_const_i32(b, tc * s->threads);
            rocke_value_t* idx_col
                = rocke_b_add(b, rocke_b_add(b, res.main_elems_per_row, c_off), s->tid_row);
            rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, s->length);
            rocke_value_t* idx_safe = rocke_b_smin(b, idx_col, s->seg_last);
            rocke_value_t* base = rocke_b_mul(b, row_n, s->HW);
            rocke_value_t* flat = rocke_b_add(b, rocke_b_add(b, base, s->hw_lo), idx_safe);
            rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, s->X, flat, s->dtype);
            rocke_value_t* dy_raw = rocke_b_load_scalar_as_f32(b, s->dY, flat, s->dtype);
            rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
            rocke_value_t* dy_m = rocke_b_select(b, valid, dy_raw, zero);
            rocke_value_t *d, *xh;
            s1 = rocke_b_fadd(b, s1, dy_m);
            d = rocke_b_fsub(b, x_raw, s->mean_c);
            xh = rocke_b_fmul(b, d, s->invstd_c);
            s2 = rocke_b_fadd(b, s2, rocke_b_fmul(b, dy_m, xh));
        }
        new_vals[0] = s1;
        new_vals[1] = s2;
        rocke_b_scf_yield(b, new_vals, 2);
    }
    rocke_b_region_leave(b);
    out[0] = for_op.op->results[0];
    out[1] = for_op.op->results[1];
}

/* dx_of(x, dy) = (a * dy - k2 * (x - mean)) - k3. */
static rocke_value_t* bwd_dx_of(rocke_ir_builder_t* b,
                                rocke_value_t* const coefs[3],
                                rocke_value_t* mean_c,
                                rocke_value_t* x,
                                rocke_value_t* dy)
{
    rocke_value_t* t_a = rocke_b_fmul(b, coefs[0], dy);
    rocke_value_t* d = rocke_b_fsub(b, x, mean_c);
    rocke_value_t* t_k2 = rocke_b_fmul(b, coefs[1], d);
    return rocke_b_fsub(b, rocke_b_fsub(b, t_a, t_k2), coefs[2]);
}

typedef struct bwd_apply_ctx
{
    const bwd_sweep_t* s;
    rocke_value_t* const* coefs;
    const rocke_tile_window_t* x_tile;
    const rocke_tile_window_t* dy_tile;
} bwd_apply_ctx_t;

static void bwd_apply_body(
    rocke_ir_builder_t* b, rocke_value_t* n_off, rocke_value_t** out, int vec, void* user)
{
    bwd_apply_ctx_t* ctx = (bwd_apply_ctx_t*)user;
    rocke_value_t* xs[8];
    rocke_value_t* dys[8];
    rocke_value_t* li[2];
    int i;

    li[0] = rocke_b_const_i32(b, 0);
    li[1] = n_off;
    rocke_tile_window_load_vec_as_f32(b, ctx->x_tile, li, 2, vec, xs);
    li[0] = rocke_b_const_i32(b, 0);
    li[1] = n_off;
    rocke_tile_window_load_vec_as_f32(b, ctx->dy_tile, li, 2, vec, dys);
    for(i = 0; i < vec; ++i)
    {
        out[i] = bwd_dx_of(b, ctx->coefs, ctx->s->mean_c, xs[i], dys[i]);
    }
}

/* _bwd_apply_rows: write dx for rows n_begin .. n_end of one channel. */
static void
    bwd_apply_rows(rocke_ir_builder_t* b, const bwd_sweep_t* s, rocke_value_t* const coefs[3])
{
    int VEC = s->vec;
    int chunk = s->threads * VEC;
    rocke_value_t *c_chunk, *trip_count, *main_elems, *step;
    rocke_for_t outer;
    char iv_n[64];
    char iv_hw[64];
    int tc;

    snprintf(iv_n, sizeof iv_n, "%s_n2", s->iv_prefix);
    snprintf(iv_hw, sizeof iv_hw, "%s_hw2", s->iv_prefix);

    c_chunk = rocke_b_const_i32(b, chunk);
    trip_count = rocke_b_div(b, s->length, c_chunk);
    c_chunk = rocke_b_const_i32(b, chunk);
    main_elems = rocke_b_mul(b, trip_count, c_chunk);

    step = rocke_b_const_i32(b, 1);
    outer = rocke_b_scf_for(b, s->n_begin, s->n_end, step, iv_n);
    rocke_b_region_enter(b, outer.body);
    {
        rocke_value_t* row_n = bwd_row_of(b, s->rows, outer.iv);
        rocke_tile_window_t x_tile, dy_tile, dx_tile;
        bwd_apply_ctx_t actx;
        int lengths2[2] = {1, 1};
        rocke_value_t* origin[2];

        origin[0] = row_n;
        origin[1] = s->hw_lo;
        rocke_make_tile_window(&x_tile, s->x_view, lengths2, origin, 2);
        rocke_make_tile_window(&dy_tile, s->dy_view, lengths2, origin, 2);
        rocke_make_tile_window(&dx_tile, s->dx_view, lengths2, origin, 2);

        actx.s = s;
        actx.coefs = coefs;
        actx.x_tile = &x_tile;
        actx.dy_tile = &dy_tile;
        rocke_pass2_row_chunks_dynamic(b,
                                       &dx_tile,
                                       s->tid_row,
                                       s->threads,
                                       VEC,
                                       main_elems,
                                       trip_count,
                                       NULL,
                                       bwd_apply_body,
                                       &actx,
                                       iv_hw);
        for(tc = 0; tc < VEC; ++tc)
        {
            rocke_value_t* c_off = rocke_b_const_i32(b, tc * s->threads);
            rocke_value_t* idx_col = rocke_b_add(b, rocke_b_add(b, main_elems, c_off), s->tid_row);
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_lt(b, idx_col, s->length));
            rocke_b_region_enter(b, iff.then_region);
            {
                rocke_value_t* base = rocke_b_mul(b, row_n, s->HW);
                rocke_value_t* flat = rocke_b_add(b, rocke_b_add(b, base, s->hw_lo), idx_col);
                rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, s->X, flat, s->dtype);
                rocke_value_t* dy_raw = rocke_b_load_scalar_as_f32(b, s->dY, flat, s->dtype);
                rocke_value_t* dx = bwd_dx_of(b, coefs, s->mean_c, x_raw, dy_raw);
                rocke_b_store_scalar_from_f32(b, s->dX, flat, dx, s->dtype);
            }
            rocke_b_region_leave(b);
        }
    }
    rocke_b_region_leave(b);
}

/* ------------------------------------------------------------------ *
 * B1: single-launch backward
 * ------------------------------------------------------------------ */

rocke_batchnorm2d_bwd_spec_t rocke_batchnorm2d_bwd_spec_default(void)
{
    rocke_batchnorm2d_bwd_spec_t s;
    s.block_size = 256;
    s.vec = 4;
    s.dtype = "f16";
    s.channels_per_block = 1;
    s.n_groups = 1;
    s.fast_reduce = false;
    s.wave_size = 64;
    s.dynamic = true;
    s.name = "rocke_batchnorm2d_bwd";
    return s;
}

rocke_status_t rocke_batchnorm2d_bwd_kernel_name(const rocke_batchnorm2d_bwd_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char part_b[32], part_v[32], part_cpb[32], part_ng[32];
    const char* parts[5];
    const char* flag_names[2] = {"fr", "dyn"};
    int flag_on[2];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(part_b, sizeof part_b, "b%d", spec->block_size);
    snprintf(part_v, sizeof part_v, "v%d", spec->vec);
    part_cpb[0] = '\0';
    part_ng[0] = '\0';
    if(spec->channels_per_block != 1)
    {
        snprintf(part_cpb, sizeof part_cpb, "cpb%d", spec->channels_per_block);
    }
    if(spec->n_groups != 1)
    {
        snprintf(part_ng, sizeof part_ng, "ng%d", spec->n_groups);
    }
    parts[0] = spec->dtype;
    parts[1] = part_b;
    parts[2] = part_v;
    parts[3] = part_cpb;
    parts[4] = part_ng;
    flag_on[0] = spec->fast_reduce ? 1 : 0;
    flag_on[1] = spec->dynamic ? 1 : 0;
    return rocke_kernel_name_join(spec->name, parts, 5, flag_names, flag_on, 2, out, out_cap, NULL);
}

/* validate_io(IOSpecRule(dtype, block_size, vec, allowed_dtypes=(f16, bf16, f32),
 * allowed_vecs=vecs)) with an arena-owned reject reason copied out. */
static bool bwd_validate_io(const char* dtype,
                            int block_size,
                            int vec,
                            const int* vecs,
                            size_t num_vecs,
                            char* reason,
                            size_t reason_cap)
{
    static const char* const kDtypes[3] = {"f16", "bf16", "f32"};
    rocke_io_spec_rule_t rule;
    rocke_arena_t arena;
    const char* why = NULL;
    int ok;

    rocke_io_spec_rule_init(&rule, dtype, block_size, vec);
    rule.allowed_dtypes = kDtypes;
    rule.num_allowed_dtypes = 3;
    rule.allowed_vecs = vecs;
    rule.num_allowed_vecs = num_vecs;
    if(rocke_arena_init(&arena, 4096) != ROCKE_OK)
    {
        bwd_set_reason(reason, reason_cap, "arena init failed");
        return false;
    }
    ok = rocke_validate_io(&arena, &rule, &why);
    if(!ok)
    {
        bwd_set_reason(reason, reason_cap, why != NULL ? why : "validate_io failed");
    }
    rocke_arena_destroy(&arena);
    return ok != 0;
}

bool rocke_batchnorm2d_is_valid_bwd_spec(const rocke_batchnorm2d_bwd_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    static const int kVecsF32[2] = {2, 4};
    static const int kVecs16[3] = {2, 4, 8};
    const rocke_archtarget_t* target;
    bool f32;

    if(reason != NULL && reason_cap > 0)
    {
        reason[0] = '\0';
    }
    if(spec == NULL)
    {
        bwd_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(!rocke_bn_wave_size_ok(spec->wave_size))
    {
        bwd_set_reason(reason, reason_cap, "wave_size must be a positive power of two");
        return false;
    }
    target = rocke_archtarget_from_gfx(arch != NULL ? arch : "gfx950");
    if(target == NULL)
    {
        bwd_set_reason(reason, reason_cap, "unknown arch");
        return false;
    }
    if(spec->channels_per_block <= 0 || spec->n_groups <= 0)
    {
        bwd_set_reason(reason, reason_cap, "channels_per_block / n_groups must be positive");
        return false;
    }
    if(spec->block_size % spec->n_groups != 0)
    {
        bwd_set_reason(reason, reason_cap, "block_size not divisible by n_groups");
        return false;
    }
    if(spec->fast_reduce && spec->block_size % spec->wave_size != 0)
    {
        bwd_set_reason(
            reason, reason_cap, "fast_reduce requires block_size divisible by wave_size");
        return false;
    }
    f32 = spec->dtype != NULL && strcmp(spec->dtype, "f32") == 0;
    if(!bwd_validate_io(spec->dtype,
                        spec->block_size,
                        spec->vec,
                        f32 ? kVecsF32 : kVecs16,
                        f32 ? 2 : 3,
                        reason,
                        reason_cap))
    {
        return false;
    }
    if(spec->block_size > rocke_archtarget_max_threads_per_block(target))
    {
        bwd_set_reason(reason, reason_cap, "block_size exceeds max_threads_per_block");
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, (long)2 * spec->block_size * 4))
    {
        bwd_set_reason(reason, reason_cap, "LDS budget exceeded");
        return false;
    }
    return true;
}

rocke_status_t
    rocke_batchnorm2d_bwd_grid(int c, const rocke_batchnorm2d_bwd_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL || spec->channels_per_block <= 0)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = c / spec->channels_per_block;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

/* Two LDS f32[BS] buffers named lds_s1 / lds_s2. */
static void bwd_lds_pair(rocke_ir_builder_t* b, int bs, rocke_value_t** a, rocke_value_t** c)
{
    rocke_tensor_view_t va, vc;
    int shape[1] = {bs};
    rocke_make_lds_view(b, &va, rocke_f32(), shape, 1, "lds_s1", NULL);
    rocke_make_lds_view(b, &vc, rocke_f32(), shape, 1, "lds_s2", NULL);
    *a = va.base;
    *c = vc.base;
}

rocke_kernel_def_t* rocke_build_batchnorm2d_bwd_dynamic(rocke_ir_builder_t* b,
                                                        const rocke_batchnorm2d_bwd_spec_t* spec)
{
    char why[256];
    const rocke_type_t* io_ty;
    int BS, CPB, NG, hw_threads, cb;
    rocke_value_t *X, *dY, *Scale, *Mean, *InvStd, *dX, *dScale, *dBias;
    rocke_value_t *p_n, *p_c, *p_h, *p_w, *HW, *tid, *base_channel;
    rocke_value_t *lds_a, *lds_c, *m_f, *hw_lo, *seg_last, *c1;
    rocke_value_t *tid_row, *lidihw, *n_trip;
    rocke_tensor_view_t x_view, dy_view, dx_view;
    rocke_stride_t row_stride[2];
    int shape2[2] = {1, 1};

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(!rocke_batchnorm2d_is_valid_bwd_spec(spec, NULL, why, sizeof why))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid batchnorm2d backward spec: %s", why);
        return NULL;
    }
    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    BS = spec->block_size;
    CPB = spec->channels_per_block;
    NG = spec->n_groups;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);
    X = rocke_bn_ptr_param(b, "X", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    dY = rocke_bn_ptr_param(b, "dY", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    Scale = rocke_bn_ptr_param(b, "Scale", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    Mean = rocke_bn_ptr_param(b, "Mean", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    InvStd = rocke_bn_ptr_param(b, "InvStd", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    dX = rocke_bn_ptr_param(b, "dX", io_ty, ROCKE_BN_PARAM_WRITEONLY, 16);
    dScale = rocke_bn_ptr_param(b, "dScale", rocke_f32(), ROCKE_BN_PARAM_WRITEONLY, 0);
    dBias = rocke_bn_ptr_param(b, "dBias", rocke_f32(), ROCKE_BN_PARAM_WRITEONLY, 0);
    p_n = rocke_b_param(b, "N", rocke_i32(), NULL);
    p_c = rocke_b_param(b, "C", rocke_i32(), NULL);
    p_h = rocke_b_param(b, "H", rocke_i32(), NULL);
    p_w = rocke_b_param(b, "W", rocke_i32(), NULL);

    HW = rocke_b_mul(b, p_h, p_w);
    tid = rocke_b_thread_id_x(b);
    base_channel = rocke_b_block_id_x(b);
    row_stride[0] = rocke_stride_value(HW);
    row_stride[1] = rocke_stride_imm(1);
    rocke_make_global_view(&x_view, X, shape2, 2, io_ty, row_stride);
    rocke_make_global_view(&dy_view, dY, shape2, 2, io_ty, row_stride);
    rocke_make_global_view(&dx_view, dX, shape2, 2, io_ty, row_stride);
    bwd_lds_pair(b, BS, &lds_a, &lds_c);
    m_f = rocke_b_sitofp_f32(b, rocke_b_mul(b, p_n, HW));
    hw_lo = rocke_b_const_i32(b, 0);
    c1 = rocke_b_const_i32(b, 1);
    seg_last = rocke_b_sub(b, HW, c1);

    hw_threads = BS / NG;
    if(NG == 1)
    {
        tid_row = tid;
        lidihw = rocke_b_const_i32(b, 0);
        n_trip = p_n;
    }
    else
    {
        tid_row = rocke_b_mod(b, tid, rocke_b_const_i32(b, hw_threads));
        lidihw = rocke_b_div(b, tid, rocke_b_const_i32(b, hw_threads));
        n_trip = rocke_b_div(b, p_n, rocke_b_const_i32(b, NG));
    }

    for(cb = 0; cb < CPB; ++cb)
    {
        rocke_value_t *channel, *mean_c, *invstd_c, *scale_c, *t1 = NULL, *t2 = NULL;
        rocke_value_t* sums[2];
        rocke_value_t* coefs[3];
        bwd_rows_t rows;
        bwd_sweep_t sw;

        if(CPB == 1)
        {
            channel = base_channel;
        }
        else
        {
            rocke_value_t* m = rocke_b_mul(b, base_channel, rocke_b_const_i32(b, CPB));
            channel = rocke_b_add(b, m, rocke_b_const_i32(b, cb));
            if(cb > 0)
            {
                rocke_b_sync(b);
            }
        }
        mean_c = rocke_b_global_load_f32(b, Mean, channel, 4);
        invstd_c = rocke_b_global_load_f32(b, InvStd, channel, 4);
        scale_c = rocke_b_global_load_f32(b, Scale, channel, 4);

        rows.ng = NG;
        rows.lidihw = lidihw;
        rows.p_c = p_c;
        rows.channel = channel;

        memset(&sw, 0, sizeof sw);
        sw.vec = spec->vec;
        sw.dtype = spec->dtype;
        sw.X = X;
        sw.dY = dY;
        sw.dX = dX;
        sw.x_view = &x_view;
        sw.dy_view = &dy_view;
        sw.dx_view = &dx_view;
        sw.HW = HW;
        sw.mean_c = mean_c;
        sw.invstd_c = invstd_c;
        sw.tid_row = tid_row;
        sw.threads = hw_threads;
        sw.n_begin = rocke_b_const_i32(b, 0);
        sw.n_end = n_trip;
        sw.rows = &rows;
        sw.hw_lo = hw_lo;
        sw.length = HW;
        sw.seg_last = seg_last;
        sw.iv_prefix = "b1";
        bwd_accumulate(b, &sw, sums);

        bwd_block_sums(b,
                       BS,
                       spec->fast_reduce,
                       spec->wave_size,
                       sums[0],
                       sums[1],
                       lds_a,
                       lds_c,
                       tid,
                       &t1,
                       &t2);
        {
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
            rocke_b_region_enter(b, iff.then_region);
            rocke_b_global_store(b, dScale, channel, t2, 1);
            rocke_b_global_store(b, dBias, channel, t1, 1);
            rocke_b_region_leave(b);
        }
        rocke_bn_bwd_coefs(b, t1, t2, scale_c, mean_c, invstd_c, m_f, coefs);
        sw.n_begin = rocke_b_const_i32(b, 0);
        bwd_apply_rows(b, &sw, coefs);
    }

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t*
    rocke_build_batchnorm2d_bwd_dynamic_new(rocke_ir_builder_t* b,
                                            const rocke_batchnorm2d_bwd_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_batchnorm2d_bwd_kernel_name(spec, name, sizeof name) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_batchnorm2d_bwd_dynamic(b, spec);
    });
}

/* ------------------------------------------------------------------ *
 * B2: split backward pair
 * ------------------------------------------------------------------ */

rocke_batchnorm2d_bwd_split_spec_t rocke_batchnorm2d_bwd_split_spec_default(void)
{
    rocke_batchnorm2d_bwd_split_spec_t s;
    s.block_size = 256;
    s.vec = 8;
    s.dtype = "f16";
    s.fast_reduce = false;
    s.save_mean_invstd = true;
    s.wave_size = 64;
    s.dynamic = true;
    s.name = "rocke_batchnorm2d_bwd_splitdyn";
    return s;
}

rocke_status_t rocke_batchnorm2d_bwd_split_kernel_name(
    const rocke_batchnorm2d_bwd_split_spec_t* spec, const char* kind, char* out, size_t out_cap)
{
    char part_b[32], part_v[32];
    const char* parts[5];
    const char* flag_names[1] = {"fr"};
    int flag_on[1];

    if(spec == NULL || out == NULL || kind == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(part_b, sizeof part_b, "b%d", spec->block_size);
    snprintf(part_v, sizeof part_v, "v%d", spec->vec);
    parts[0] = spec->dtype;
    parts[1] = part_b;
    parts[2] = part_v;
    parts[3] = kind;
    parts[4] = "dyn";
    flag_on[0] = spec->fast_reduce ? 1 : 0;
    return rocke_kernel_name_join(spec->name, parts, 5, flag_names, flag_on, 1, out, out_cap, NULL);
}

bool rocke_batchnorm2d_is_valid_bwd_split_spec(const rocke_batchnorm2d_bwd_split_spec_t* spec,
                                               const char* arch,
                                               char* reason,
                                               size_t reason_cap)
{
    /* Python applies is_valid_split_dyn_spec to the backward split spec. */
    rocke_batchnorm2d_split_dyn_spec_t fwd;
    if(spec == NULL)
    {
        bwd_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    fwd = rocke_batchnorm2d_split_dyn_spec_default();
    fwd.block_size = spec->block_size;
    fwd.vec = spec->vec;
    fwd.dtype = spec->dtype;
    fwd.fast_reduce = spec->fast_reduce;
    fwd.wave_size = spec->wave_size;
    return rocke_batchnorm2d_is_valid_split_dyn_spec(&fwd, arch, reason, reason_cap);
}

int rocke_batchnorm2d_bwd_split_workspace_floats(int c, int splits)
{
    return c * splits * 2;
}

/* Common prologue of both split kernels after the params. */
typedef struct bwd_split_common
{
    rocke_value_t *HW, *tid, *channel;
    rocke_bn_split_geom_t g;
    rocke_tensor_view_t x_view, dy_view, dx_view;
    rocke_value_t *lds_a, *lds_c;
} bwd_split_common_t;

static void bwd_split_prologue(rocke_ir_builder_t* b,
                               const rocke_type_t* io_ty,
                               int bs,
                               rocke_value_t* X,
                               rocke_value_t* dY,
                               rocke_value_t* dX,
                               rocke_value_t* p_n,
                               rocke_value_t* p_h,
                               rocke_value_t* p_w,
                               rocke_value_t* p_nch,
                               rocke_value_t* p_hwch,
                               bwd_split_common_t* cm)
{
    rocke_stride_t row_stride[2];
    int shape2[2] = {1, 1};
    cm->HW = rocke_b_mul(b, p_h, p_w);
    cm->tid = rocke_b_thread_id_x(b);
    cm->channel = rocke_b_block_id_x(b);
    rocke_bn_split_dyn_geometry(b, p_n, p_nch, p_hwch, cm->HW, &cm->g);
    row_stride[0] = rocke_stride_value(cm->HW);
    row_stride[1] = rocke_stride_imm(1);
    rocke_make_global_view(&cm->x_view, X, shape2, 2, io_ty, row_stride);
    rocke_make_global_view(&cm->dy_view, dY, shape2, 2, io_ty, row_stride);
    if(dX != NULL)
    {
        rocke_make_global_view(&cm->dx_view, dX, shape2, 2, io_ty, row_stride);
    }
    bwd_lds_pair(b, bs, &cm->lds_a, &cm->lds_c);
}

rocke_kernel_def_t*
    rocke_build_batchnorm2d_bwd_split_reduce(rocke_ir_builder_t* b,
                                             const rocke_batchnorm2d_bwd_split_spec_t* spec)
{
    char why[256];
    const rocke_type_t* io_ty;
    int BS;
    rocke_value_t *X, *dY, *Mean, *InvStd, *Workspace;
    rocke_value_t *p_n, *p_c, *p_h, *p_w, *p_nch, *p_hwch, *p_sh, *p_sn;
    rocke_value_t *seg_last, *t, *c, *mean_c, *invstd_c, *t1 = NULL, *t2 = NULL;
    rocke_value_t* sums[2];
    bwd_split_common_t cm;
    bwd_rows_t rows;
    bwd_sweep_t sw;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(!rocke_batchnorm2d_is_valid_bwd_split_spec(spec, NULL, why, sizeof why))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid batchnorm2d backward split spec: %s", why);
        return NULL;
    }
    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    BS = spec->block_size;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);
    X = rocke_bn_ptr_param(b, "X", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    dY = rocke_bn_ptr_param(b, "dY", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    Mean = rocke_bn_ptr_param(b, "Mean", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    InvStd = rocke_bn_ptr_param(b, "InvStd", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    Workspace = rocke_bn_ptr_param(b, "Workspace", rocke_f32(), ROCKE_BN_PARAM_PLAIN, 0);
    p_n = rocke_b_param(b, "N", rocke_i32(), NULL);
    p_c = rocke_b_param(b, "C", rocke_i32(), NULL);
    p_h = rocke_b_param(b, "H", rocke_i32(), NULL);
    p_w = rocke_b_param(b, "W", rocke_i32(), NULL);
    p_nch = rocke_b_param(b, "NChunk", rocke_i32(), NULL);
    p_hwch = rocke_b_param(b, "HwChunk", rocke_i32(), NULL);
    p_sh = rocke_b_param(b, "SplitHw", rocke_i32(), NULL);
    p_sn = rocke_b_param(b, "SplitN", rocke_i32(), NULL);

    bwd_split_prologue(b, io_ty, BS, X, dY, NULL, p_n, p_h, p_w, p_nch, p_hwch, &cm);
    c = rocke_b_const_i32(b, 1);
    t = rocke_b_sub(b, cm.g.seg_len, c);
    c = rocke_b_const_i32(b, 0);
    seg_last = rocke_b_smax(b, t, c);
    mean_c = rocke_b_global_load_f32(b, Mean, cm.channel, 4);
    invstd_c = rocke_b_global_load_f32(b, InvStd, cm.channel, 4);

    rows.ng = 1;
    rows.lidihw = NULL;
    rows.p_c = p_c;
    rows.channel = cm.channel;
    memset(&sw, 0, sizeof sw);
    sw.vec = spec->vec;
    sw.dtype = spec->dtype;
    sw.X = X;
    sw.dY = dY;
    sw.x_view = &cm.x_view;
    sw.dy_view = &cm.dy_view;
    sw.HW = cm.HW;
    sw.mean_c = mean_c;
    sw.invstd_c = invstd_c;
    sw.tid_row = cm.tid;
    sw.threads = BS;
    sw.n_begin = cm.g.n_lo;
    sw.n_end = cm.g.n_hi;
    sw.rows = &rows;
    sw.hw_lo = cm.g.hw_lo;
    sw.length = cm.g.seg_len;
    sw.seg_last = seg_last;
    sw.iv_prefix = "r";
    bwd_accumulate(b, &sw, sums);
    bwd_block_sums(b,
                   BS,
                   spec->fast_reduce,
                   spec->wave_size,
                   sums[0],
                   sums[1],
                   cm.lds_a,
                   cm.lds_c,
                   cm.tid,
                   &t1,
                   &t2);
    {
        rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, cm.tid, rocke_b_const_i32(b, 0)));
        rocke_b_region_enter(b, iff.then_region);
        {
            rocke_value_t* split_idx = rocke_b_add(b, rocke_b_mul(b, cm.g.sn_id, p_sh), cm.g.sh_id);
            rocke_value_t* splits = rocke_b_mul(b, p_sh, p_sn);
            rocke_value_t* base = rocke_b_add(b, rocke_b_mul(b, cm.channel, splits), split_idx);
            rocke_value_t* slot = rocke_b_mul(b, base, rocke_b_const_i32(b, 2));
            rocke_b_global_store(b, Workspace, slot, t1, 1);
            rocke_b_global_store(
                b, Workspace, rocke_b_add(b, slot, rocke_b_const_i32(b, 1)), t2, 1);
        }
        rocke_b_region_leave(b);
    }
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t*
    rocke_build_batchnorm2d_bwd_split_apply(rocke_ir_builder_t* b,
                                            const rocke_batchnorm2d_bwd_split_spec_t* spec)
{
    char why[256];
    const rocke_type_t* io_ty;
    int BS;
    rocke_value_t *X, *dY, *Scale, *Mean, *InvStd, *Workspace, *dX, *dScale, *dBias;
    rocke_value_t *p_n, *p_c, *p_h, *p_w, *p_nch, *p_hwch, *p_sh, *p_sn;
    rocke_value_t *splits, *has_slot, *slot_tid, *slot, *ld, *p1, *p2, *t1 = NULL, *t2 = NULL;
    rocke_value_t *mean_c, *invstd_c, *scale_c, *owner, *e0, *e1, *m_f;
    rocke_value_t* coefs[3];
    bwd_split_common_t cm;
    bwd_rows_t rows;
    bwd_sweep_t sw;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(!rocke_batchnorm2d_is_valid_bwd_split_spec(spec, NULL, why, sizeof why))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid batchnorm2d backward split spec: %s", why);
        return NULL;
    }
    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    BS = spec->block_size;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);
    X = rocke_bn_ptr_param(b, "X", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    dY = rocke_bn_ptr_param(b, "dY", io_ty, ROCKE_BN_PARAM_READONLY, 16);
    Scale = rocke_bn_ptr_param(b, "Scale", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    Mean = rocke_bn_ptr_param(b, "Mean", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    InvStd = rocke_bn_ptr_param(b, "InvStd", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    Workspace = rocke_bn_ptr_param(b, "Workspace", rocke_f32(), ROCKE_BN_PARAM_READONLY, 0);
    dX = rocke_bn_ptr_param(b, "dX", io_ty, ROCKE_BN_PARAM_WRITEONLY, 16);
    dScale = rocke_bn_ptr_param(b, "dScale", rocke_f32(), ROCKE_BN_PARAM_WRITEONLY, 0);
    dBias = rocke_bn_ptr_param(b, "dBias", rocke_f32(), ROCKE_BN_PARAM_WRITEONLY, 0);
    p_n = rocke_b_param(b, "N", rocke_i32(), NULL);
    p_c = rocke_b_param(b, "C", rocke_i32(), NULL);
    p_h = rocke_b_param(b, "H", rocke_i32(), NULL);
    p_w = rocke_b_param(b, "W", rocke_i32(), NULL);
    p_nch = rocke_b_param(b, "NChunk", rocke_i32(), NULL);
    p_hwch = rocke_b_param(b, "HwChunk", rocke_i32(), NULL);
    p_sh = rocke_b_param(b, "SplitHw", rocke_i32(), NULL);
    p_sn = rocke_b_param(b, "SplitN", rocke_i32(), NULL);

    bwd_split_prologue(b, io_ty, BS, X, dY, dX, p_n, p_h, p_w, p_nch, p_hwch, &cm);

    splits = rocke_b_mul(b, p_sh, p_sn);
    has_slot = rocke_b_cmp_lt(b, cm.tid, splits);
    slot_tid = rocke_b_select(b, has_slot, cm.tid, rocke_b_const_i32(b, 0));
    slot = rocke_b_add(b, rocke_b_mul(b, cm.channel, splits), slot_tid);
    slot = rocke_b_mul(b, slot, rocke_b_const_i32(b, 2));
    ld = rocke_b_global_load_f32(b, Workspace, slot, 4);
    p1 = rocke_b_select(b, has_slot, ld, rocke_b_const_f32(b, 0.0));
    ld = rocke_b_global_load_f32(b, Workspace, rocke_b_add(b, slot, rocke_b_const_i32(b, 1)), 4);
    p2 = rocke_b_select(b, has_slot, ld, rocke_b_const_f32(b, 0.0));
    bwd_block_sums(
        b, BS, spec->fast_reduce, spec->wave_size, p1, p2, cm.lds_a, cm.lds_c, cm.tid, &t1, &t2);

    mean_c = rocke_b_global_load_f32(b, Mean, cm.channel, 4);
    invstd_c = rocke_b_global_load_f32(b, InvStd, cm.channel, 4);
    scale_c = rocke_b_global_load_f32(b, Scale, cm.channel, 4);
    e0 = rocke_b_cmp_eq(b, cm.g.sh_id, rocke_b_const_i32(b, 0));
    e1 = rocke_b_cmp_eq(b, cm.g.sn_id, rocke_b_const_i32(b, 0));
    owner = rocke_b_land(b, e0, e1);
    e0 = rocke_b_cmp_eq(b, cm.tid, rocke_b_const_i32(b, 0));
    owner = rocke_b_land(b, owner, e0);
    {
        rocke_if_t iff = rocke_b_scf_if(b, owner);
        rocke_b_region_enter(b, iff.then_region);
        rocke_b_global_store(b, dScale, cm.channel, t2, 1);
        rocke_b_global_store(b, dBias, cm.channel, t1, 1);
        rocke_b_region_leave(b);
    }
    m_f = rocke_b_sitofp_f32(b, rocke_b_mul(b, p_n, cm.HW));
    rocke_bn_bwd_coefs(b, t1, t2, scale_c, mean_c, invstd_c, m_f, coefs);

    rows.ng = 1;
    rows.lidihw = NULL;
    rows.p_c = p_c;
    rows.channel = cm.channel;
    memset(&sw, 0, sizeof sw);
    sw.vec = spec->vec;
    sw.dtype = spec->dtype;
    sw.X = X;
    sw.dY = dY;
    sw.dX = dX;
    sw.x_view = &cm.x_view;
    sw.dy_view = &cm.dy_view;
    sw.dx_view = &cm.dx_view;
    sw.HW = cm.HW;
    sw.mean_c = mean_c;
    sw.tid_row = cm.tid;
    sw.threads = BS;
    sw.n_begin = cm.g.n_lo;
    sw.n_end = cm.g.n_hi;
    sw.rows = &rows;
    sw.hw_lo = cm.g.hw_lo;
    sw.length = cm.g.seg_len;
    sw.iv_prefix = "a";
    bwd_apply_rows(b, &sw, coefs);

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

static rocke_kernel_def_t* bwd_split_new(rocke_ir_builder_t* b,
                                         const rocke_batchnorm2d_bwd_split_spec_t* spec,
                                         bool reduce)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_batchnorm2d_bwd_split_kernel_name(
               spec, reduce ? "reduce" : "apply", name, sizeof name)
           != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return reduce ? rocke_build_batchnorm2d_bwd_split_reduce(b, spec)
                      : rocke_build_batchnorm2d_bwd_split_apply(b, spec);
    });
}

rocke_kernel_def_t*
    rocke_build_batchnorm2d_bwd_split_reduce_new(rocke_ir_builder_t* b,
                                                 const rocke_batchnorm2d_bwd_split_spec_t* spec)
{
    return bwd_split_new(b, spec, true);
}

rocke_kernel_def_t*
    rocke_build_batchnorm2d_bwd_split_apply_new(rocke_ir_builder_t* b,
                                                const rocke_batchnorm2d_bwd_split_spec_t* spec)
{
    return bwd_split_new(b, spec, false);
}
