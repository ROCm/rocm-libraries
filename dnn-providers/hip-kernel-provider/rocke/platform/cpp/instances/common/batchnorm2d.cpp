// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_batchnorm2d.cpp -- C99 port of
 * rocke/instances/common/batchnorm2d.py.
 *
 * Byte-faithful reproduction of build_batchnorm2d()'s builder-call sequence.
 * See instance_batchnorm2d.h for the public surface.
 */

#include "rocke/instance_batchnorm2d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.sweep.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

/* ------------------------------------------------------------------ *
 * spec defaults / properties
 * ------------------------------------------------------------------ */

rocke_batchnorm2d_spec_t rocke_batchnorm2d_spec_default(void)
{
    rocke_batchnorm2d_spec_t s;
    s.n = 0; /* required: caller must set */
    s.c = 0; /* required: caller must set */
    s.h = 0; /* required: caller must set */
    s.w = 0; /* required: caller must set */
    s.block_size = 256;
    s.vec = 4;
    s.dtype = "f16";
    s.training = true;
    s.save_mean_invstd = false;
    s.channels_per_block = 1;
    s.n_groups = 1;
    s.fast_reduce = false;
    s.wave_size = 64;
    s.dynamic = false;
    s.name = "rocke_batchnorm2d_fwd";
    return s;
}

int rocke_batchnorm2d_hw(const rocke_batchnorm2d_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->h * spec->w;
}

int rocke_batchnorm2d_n_per_channel(const rocke_batchnorm2d_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->n * spec->h * spec->w;
}

int rocke_batchnorm2d_elems_per_thread(const rocke_batchnorm2d_spec_t* spec)
{
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return rocke_batchnorm2d_n_per_channel(spec) / spec->block_size;
}

/* BatchNorm2DSpec.kernel_name():
 *   kernel_name_join(self.name, self.dtype, f"N{n}C{c}H{h}W{w}", f"b{block_size}",
 *                    f"v{vec}",
 *                    f"cpb{channels_per_block}" if channels_per_block != 1 else "",
 *                    f"ng{n_groups}" if n_groups != 1 else "",
 *                    flags={"train": training, "smv": save_mean_invstd})
 * kernel_name_join drops empty-string parts, so empty part_cpb/part_ng (the
 * channels_per_block==1 / n_groups==1 defaults) reproduce the pre-lever name
 * exactly. */
rocke_status_t rocke_batchnorm2d_kernel_name(const rocke_batchnorm2d_spec_t* spec,
                                             char* out,
                                             size_t out_cap)
{
    char part_dims[64];
    char part_b[32];
    char part_v[32];
    char part_cpb[32];
    char part_ng[32];
    const char* parts[6];
    const char* flag_names[4];
    int flag_on[4];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    snprintf(part_dims, sizeof part_dims, "N%dC%dH%dW%d", spec->n, spec->c, spec->h, spec->w);
    snprintf(part_b, sizeof part_b, "b%d", spec->block_size);
    snprintf(part_v, sizeof part_v, "v%d", spec->vec);
    if(spec->channels_per_block != 1)
    {
        snprintf(part_cpb, sizeof part_cpb, "cpb%d", spec->channels_per_block);
    }
    else
    {
        part_cpb[0] = '\0';
    }
    if(spec->n_groups != 1)
    {
        snprintf(part_ng, sizeof part_ng, "ng%d", spec->n_groups);
    }
    else
    {
        part_ng[0] = '\0';
    }

    parts[0] = spec->dtype;
    parts[1] = part_dims;
    parts[2] = part_b;
    parts[3] = part_v;
    parts[4] = part_cpb;
    parts[5] = part_ng;

    flag_names[0] = "train";
    flag_on[0] = spec->training ? 1 : 0;
    flag_names[1] = "smv";
    flag_on[1] = spec->save_mean_invstd ? 1 : 0;
    flag_names[2] = "fr";
    flag_on[2] = spec->fast_reduce ? 1 : 0;
    flag_names[3] = "dyn";
    flag_on[3] = spec->dynamic ? 1 : 0;

    return rocke_kernel_name_join(spec->name, parts, 6, flag_names, flag_on, 4, out, out_cap, NULL);
}

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ */

static void bn_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

bool rocke_batchnorm2d_is_valid_spec(const rocke_batchnorm2d_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap)
{
    const rocke_archtarget_t* target;
    rocke_io_spec_rule_t rule;
    rocke_arena_t arena;
    const char* why = NULL;
    int io_ok;
    int elems;
    int max_thr;
    long bytes_lds;
    bool two_pass;

    if(reason != NULL && reason_cap > 0)
    {
        reason[0] = '\0';
    }
    if(spec == NULL)
    {
        bn_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        bn_set_reason(reason, reason_cap, "unknown arch");
        return false;
    }

    if(spec->n <= 0 || spec->c <= 0 || spec->h <= 0 || spec->w <= 0)
    {
        bn_set_reason(reason, reason_cap, "n, c, h, w must all be positive");
        return false;
    }

    if(spec->save_mean_invstd && !spec->training)
    {
        bn_set_reason(reason, reason_cap, "save_mean_invstd requires training=True");
        return false;
    }

    if(spec->channels_per_block <= 0)
    {
        bn_set_reason(reason, reason_cap, "channels_per_block must be positive");
        return false;
    }
    if(!spec->dynamic && spec->c % spec->channels_per_block != 0)
    {
        bn_set_reason(reason, reason_cap, "c not divisible by channels_per_block");
        return false;
    }
    /* In dynamic mode spec->c is a capacity placeholder, not the real
     * shape: CPB needs no kernel-body change (C is never read in-kernel,
     * only for host-side grid sizing), so "CPB divides C" moves from a
     * build-time check here to a dispatch-time admission check against
     * the REAL runtime C. */

    if(spec->n_groups <= 0)
    {
        bn_set_reason(reason, reason_cap, "n_groups must be positive");
        return false;
    }
    if(spec->block_size % spec->n_groups != 0)
    {
        bn_set_reason(reason, reason_cap, "block_size not divisible by n_groups");
        return false;
    }
    if(!spec->dynamic && spec->n % spec->n_groups != 0)
    {
        bn_set_reason(reason, reason_cap, "n not divisible by n_groups");
        return false;
    }


    if(spec->fast_reduce && !spec->training)
    {
        bn_set_reason(reason, reason_cap, "fast_reduce requires training=True");
        return false;
    }
    if(spec->fast_reduce && spec->block_size % spec->wave_size != 0)
    {
        bn_set_reason(reason, reason_cap, "fast_reduce requires block_size divisible by wave_size");
        return false;
    }

    /* block_size is the real launch parameter (one CTA's thread count), so
     * it alone -- not hw_threads = block_size/n_groups -- is what must come
     * from the hardware-meaningful allowed-block-sizes set; dtype/vec are
     * independent of n_groups. n_groups==1's row sweep handles a
     * non-divisible HW with an in-kernel scalar tail (rocke_build_batchnorm2d's
     * tail_chunks block); n_groups>1 reuses the SAME masked-tail idiom
     * scoped to hw_threads (that function's tail_chunks_ng block), so
     * hw_threads is a kernel-internal sub-group thread count, not a real
     * launch parameter -- no n_per_block divisibility check needed for it
     * either. */
    rocke_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec);
    rule.n_per_block_set = 0;
    rule.max_elems_per_thread_set = 0; /* checked separately below */

    if(rocke_arena_init(&arena, 4096) != ROCKE_OK)
    {
        bn_set_reason(reason, reason_cap, "arena init failed");
        return false;
    }
    io_ok = rocke_validate_io(&arena, &rule, &why);
    if(!io_ok)
    {
        bn_set_reason(reason, reason_cap, why != NULL ? why : "validate_io failed");
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    if(!spec->dynamic)
    {
        /* The dynamic path always streams X twice (no per-sample register
         * cache), so it carries no VGPR cap; n/h/w are capacity
         * placeholders there, not the real shape. */
        elems = rocke_batchnorm2d_elems_per_thread(spec);
        two_pass = rocke_row_norm_needs_two_pass(elems, ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD);
        if(!two_pass && elems > ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD)
        {
            bn_set_reason(reason, reason_cap, "elems_per_thread exceeds register-tile cap");
            return false;
        }
    }

    max_thr = rocke_archtarget_max_threads_per_block(target);
    if(spec->block_size > max_thr)
    {
        bn_set_reason(reason, reason_cap, "block_size > max_threads_per_block");
        return false;
    }

    if(spec->training)
    {
        bytes_lds = (long)3 * (long)spec->block_size * 4L;
        if(!rocke_archtarget_fits_lds(target, bytes_lds))
        {
            bn_set_reason(reason, reason_cap, "LDS budget exceeds cap");
            return false;
        }
    }

    return true;
}

/* ------------------------------------------------------------------ *
 * tree_reduce combiner cookie: forward b.fadd as a rocke_combine_fn.
 * ------------------------------------------------------------------ */

static rocke_value_t*
    bn_fadd_combine(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, void* user)
{
    (void)user;
    return rocke_b_fadd(b, a, c);
}

/* ------------------------------------------------------------------ *
 * pass-1 / pass-2 body contexts (Python closures over nonlocal state).
 * ------------------------------------------------------------------ */

typedef struct bn_pass1_ctx
{
    rocke_value_t* sum_p;
    rocke_value_t* sumsq_p;
} bn_pass1_ctx_t;

static void bn_pass1_body(rocke_ir_builder_t* b,
                          rocke_value_t* n_off,
                          rocke_value_t* const* x_scalars,
                          int vec,
                          void* user)
{
    bn_pass1_ctx_t* ctx = (bn_pass1_ctx_t*)user;
    rocke_value_t* sq[8];
    rocke_value_t* xs[8];
    rocke_value_t* part_sum;
    rocke_value_t* part_sumsq;
    int i;

    (void)n_off;
    if(vec > 8)
    {
        return;
    }

    for(i = 0; i < vec; ++i)
    {
        xs[i] = x_scalars[i];
        sq[i] = rocke_b_fmul(b, x_scalars[i], x_scalars[i]);
    }

    part_sum = rocke_tree_reduce(b, bn_fadd_combine, NULL, xs, vec);
    ctx->sum_p = rocke_b_fadd(b, ctx->sum_p, part_sum);

    part_sumsq = rocke_tree_reduce(b, bn_fadd_combine, NULL, sq, vec);
    ctx->sumsq_p = rocke_b_fadd(b, ctx->sumsq_p, part_sumsq);
}

typedef struct bn_pass2_ctx
{
    bool two_pass;
    bool training;
    int vec;
    const rocke_tile_window_t* x_tile_n;
    rocke_value_t* mean;
    rocke_value_t* scale; /* inv_std * gamma */
    rocke_value_t* beta_v;
} bn_pass2_ctx_t;

static void bn_pass2_body(rocke_ir_builder_t* b,
                          rocke_value_t* n_off,
                          int k,
                          rocke_value_t* const* x_scalars,
                          int num_x,
                          rocke_value_t** out,
                          int vec,
                          void* user)
{
    bn_pass2_ctx_t* ctx = (bn_pass2_ctx_t*)user;
    rocke_value_t* xs[8];
    rocke_value_t* li[2];
    int i;

    (void)k;
    if(vec > 8)
    {
        return;
    }

    if(ctx->two_pass || !ctx->training)
    {
        li[0] = rocke_b_const_i32(b, 0);
        li[1] = n_off;
        rocke_tile_window_load_vec_as_f32(b, ctx->x_tile_n, li, 2, vec, xs);
    }
    else
    {
        for(i = 0; i < vec && i < num_x; ++i)
        {
            xs[i] = x_scalars[i];
        }
    }

    for(i = 0; i < vec; ++i)
    {
        rocke_value_t* dx = rocke_b_fsub(b, xs[i], ctx->mean);
        out[i] = rocke_b_fadd(b, rocke_b_fmul(b, dx, ctx->scale), ctx->beta_v);
    }
}

/* ------------------------------------------------------------------ *
 * build_batchnorm2d_dynamic -- AOT-generic variant (runtime N/H/W)
 *
 * C mirror of Python's _build_batchnorm2d_dynamic: see that function's
 * docstring. Requires channels_per_block == n_groups == 1 (enforced by
 * is_valid_spec). The N-loop is a genuine scf.for (rocke_b_scf_for_iter)
 * carrying the running sum/sumsq/tail-count accumulators; nested inside
 * it, the per-sample H*W sweep is itself a runtime loop
 * (rocke_sweep_row_chunks_dynamic) plus a vec-wide masked scalar tail.
 * ------------------------------------------------------------------ */

static void bn_dyn_pass1_body(rocke_ir_builder_t* b,
                              rocke_value_t* n_off,
                              rocke_value_t* const* x_scalars,
                              int vec,
                              rocke_value_t* const* iter_vars,
                              int num_iter_vars,
                              rocke_value_t** new_iter_vals,
                              void* user)
{
    rocke_value_t* sq[8];
    rocke_value_t* part_sum;
    rocke_value_t* part_sumsq;
    int i;
    (void)n_off;
    (void)num_iter_vars;
    (void)user;

    for(i = 0; i < vec; ++i)
    {
        sq[i] = rocke_b_fmul(b, x_scalars[i], x_scalars[i]);
    }
    part_sum = rocke_tree_reduce(b, bn_fadd_combine, NULL, x_scalars, vec);
    new_iter_vals[0] = rocke_b_fadd(b, iter_vars[0], part_sum);
    part_sumsq = rocke_tree_reduce(b, bn_fadd_combine, NULL, sq, vec);
    new_iter_vals[1] = rocke_b_fadd(b, iter_vars[1], part_sumsq);
}

typedef struct bn_dyn_pass2_ctx
{
    rocke_value_t* mean;
    rocke_value_t* scale;
    rocke_value_t* beta_v;
    const rocke_tile_window_t* x_tile;
} bn_dyn_pass2_ctx_t;

static void bn_dyn_pass2_body(rocke_ir_builder_t* b,
                              rocke_value_t* n_off,
                              rocke_value_t** out,
                              int vec,
                              void* user)
{
    bn_dyn_pass2_ctx_t* ctx = (bn_dyn_pass2_ctx_t*)user;
    rocke_value_t* xs[8];
    rocke_value_t* li[2];
    int i;

    li[0] = rocke_b_const_i32(b, 0);
    li[1] = n_off;
    rocke_tile_window_load_vec_as_f32(b, ctx->x_tile, li, 2, vec, xs);
    for(i = 0; i < vec; ++i)
    {
        rocke_value_t* dx = rocke_b_fsub(b, xs[i], ctx->mean);
        out[i] = rocke_b_fadd(b, rocke_b_fmul(b, dx, ctx->scale), ctx->beta_v);
    }
}

static rocke_kernel_def_t* rocke_build_batchnorm2d_dynamic(rocke_ir_builder_t* b,
                                                            const rocke_batchnorm2d_spec_t* spec)
{
    const rocke_type_t* io_ty;
    int BS, VEC, CAP;
    rocke_value_t *X, *Gamma, *Beta, *Y, *RunningMean, *RunningVar;
    rocke_value_t *Mean = NULL, *InvStd = NULL;
    rocke_value_t *p_n, *p_c, *p_h, *p_w, *eps, *momentum;
    rocke_value_t *HW, *tid, *base_channel, *channel;
    int CPB, NG;
    int cb;
    rocke_tensor_view_t x_view, y_view;
    rocke_tensor_view_t lds_mean_v, lds_m2_v, lds_count_v;
    rocke_value_t *lds_mean = NULL, *lds_m2 = NULL, *lds_count = NULL;
    rocke_value_t *mean = NULL, *var = NULL, *inv_std, *gamma_v, *beta_v, *scale;
    rocke_stride_t row_stride[2];
    int shape2[2];
    int ldsshape[1];
    int lengths2[2];
    rocke_value_t* origin[2];
    rocke_param_opts_t opts;
    int tc;

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
        return NULL;
    BS = spec->block_size;
    VEC = spec->vec;
    CPB = spec->channels_per_block;
    NG = spec->n_groups;
    CAP = spec->n * spec->c;
    if(CAP <= 0)
        CAP = 1;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    X = rocke_b_param(b, "X", rocke_ptr_type(b, io_ty, "global"), &opts);
    Gamma = rocke_b_param(b, "Gamma", rocke_ptr_type(b, io_ty, "global"), &opts);
    Beta = rocke_b_param(b, "Beta", rocke_ptr_type(b, io_ty, "global"), &opts);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    Y = rocke_b_param(b, "Y", rocke_ptr_type(b, io_ty, "global"), &opts);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    RunningMean = rocke_b_param(b, "RunningMean", rocke_ptr_type(b, rocke_f32(), "global"), &opts);
    RunningVar = rocke_b_param(b, "RunningVar", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    if(spec->save_mean_invstd)
    {
        memset(&opts, 0, sizeof opts);
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        Mean = rocke_b_param(b, "Mean", rocke_ptr_type(b, io_ty, "global"), &opts);
        InvStd = rocke_b_param(b, "InvStd", rocke_ptr_type(b, io_ty, "global"), &opts);
    }

    p_n = rocke_b_param(b, "N", rocke_i32(), NULL);
    p_c = rocke_b_param(b, "C", rocke_i32(), NULL);
    p_h = rocke_b_param(b, "H", rocke_i32(), NULL);
    p_w = rocke_b_param(b, "W", rocke_i32(), NULL);
    eps = rocke_b_param(b, "eps", rocke_f32(), NULL);
    momentum = rocke_b_param(b, "momentum", rocke_f32(), NULL);

    HW = rocke_b_mul(b, p_h, p_w);
    tid = rocke_b_thread_id_x(b);
    base_channel = rocke_b_block_id_x(b);

    shape2[0] = CAP;
    shape2[1] = CAP;
    row_stride[0] = rocke_stride_value(HW);
    row_stride[1] = rocke_stride_imm(1);
    rocke_make_global_view(&x_view, X, shape2, 2, io_ty, row_stride);
    rocke_make_global_view(&y_view, Y, shape2, 2, io_ty, row_stride);

    if(spec->training)
    {
        ldsshape[0] = BS;
        rocke_make_lds_view(b, &lds_mean_v, rocke_f32(), ldsshape, 1, "lds_mean", NULL);
        rocke_make_lds_view(b, &lds_m2_v, rocke_f32(), ldsshape, 1, "lds_m2", NULL);
        rocke_make_lds_view(b, &lds_count_v, rocke_f32(), ldsshape, 1, "lds_count", NULL);
        lds_mean = lds_mean_v.base;
        lds_m2 = lds_m2_v.base;
        lds_count = lds_count_v.base;
    }

    /* CPB channels, one fully resolved at a time -- same shape as
     * build_batchnorm2d's own CPB loop, including the "every lane done
     * reading channel cb's LDS before any lane writes channel cb+1's LDS"
     * sync. CPB==1 collapses to exactly the pre-CPB-support emission. */
    for(cb = 0; cb < CPB; ++cb)
    {
        if(CPB == 1)
        {
            channel = base_channel;
        }
        else
        {
            {
                rocke_value_t* cb_mul = rocke_b_mul(b, base_channel, rocke_b_const_i32(b, CPB));
                rocke_value_t* cb_const = rocke_b_const_i32(b, cb);
                channel = rocke_b_add(b, cb_mul, cb_const);
            }
            if(spec->training && cb > 0)
            {
                rocke_b_sync(b);
            }
        }

    if(NG == 1)
    {
    if(spec->training)
    {
        rocke_iter_arg_t outer_iter_args[3];
        rocke_for_t outer_for;
        rocke_value_t *sum_p, *sumsq_p, *tail_count_p;
        rocke_value_t *trip_count, *main_per_thread_total, *count_val;
        rocke_value_t *mean_p, *m2_p;

        outer_iter_args[0].name = "sum_p";
        outer_iter_args[0].init = rocke_b_const_f32(b, 0.0);
        outer_iter_args[1].name = "sumsq_p";
        outer_iter_args[1].init = rocke_b_const_f32(b, 0.0);
        outer_iter_args[2].name = "tail_count_p";
        outer_iter_args[2].init = rocke_b_const_f32(b, 0.0);

        outer_for = rocke_b_scf_for_iter(
            b, rocke_b_const_i32(b, 0), p_n, rocke_b_const_i32(b, 1), outer_iter_args, 3, "n0", false, true);
        rocke_b_region_enter(b, outer_for.body);
        {
            rocke_value_t* n_iv = outer_for.iv;
            rocke_value_t* row_n = rocke_b_add(b, rocke_b_mul(b, n_iv, p_c), channel);
            rocke_tile_window_t x_tile_n;
            rocke_iter_arg_t inner_iter_args[2];
            rocke_sweep_dynamic_result_t sweep_res;
            rocke_value_t *sp, *sqp, *tcp;
            rocke_value_t* new_vals[3];

            lengths2[0] = 1;
            lengths2[1] = 1;
            origin[0] = row_n;
            origin[1] = rocke_b_const_i32(b, 0);
            rocke_make_tile_window(&x_tile_n, &x_view, lengths2, origin, 2);

            inner_iter_args[0].name = "sum_p_hw";
            inner_iter_args[0].init = outer_for.iter_vars[0];
            inner_iter_args[1].name = "sumsq_p_hw";
            inner_iter_args[1].init = outer_for.iter_vars[1];

            sweep_res = rocke_sweep_row_chunks_dynamic(b,
                                                       &x_tile_n,
                                                       tid,
                                                       BS,
                                                       VEC,
                                                       HW,
                                                       inner_iter_args,
                                                       2,
                                                       NULL,
                                                       bn_dyn_pass1_body,
                                                       NULL,
                                                       "hw0");
            sp = sweep_res.final_iter_vals[0];
            sqp = sweep_res.final_iter_vals[1];
            tcp = outer_for.iter_vars[2];

            for(tc = 0; tc < VEC; ++tc)
            {
                rocke_value_t* idx_col = rocke_b_add(
                    b,
                    rocke_b_add(b, sweep_res.main_elems_per_row, rocke_b_const_i32(b, tc * BS)),
                    tid);
                rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, HW);
                rocke_value_t* idx_safe =
                    rocke_b_smin(b, idx_col, rocke_b_sub(b, HW, rocke_b_const_i32(b, 1)));
                rocke_value_t* flat = rocke_b_add(b, rocke_b_mul(b, row_n, HW), idx_safe);
                rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                rocke_value_t* zero_f = rocke_b_const_f32(b, 0.0);
                rocke_value_t* x_masked = rocke_b_select(b, valid, x_raw, zero_f);
                rocke_value_t* one_f2 = rocke_b_const_f32(b, 1.0);
                rocke_value_t* zero_f2 = rocke_b_const_f32(b, 0.0);
                rocke_value_t* valid_f = rocke_b_select(b, valid, one_f2, zero_f2);
                sp = rocke_b_fadd(b, sp, x_masked);
                sqp = rocke_b_fadd(b, sqp, rocke_b_fmul(b, x_masked, x_masked));
                tcp = rocke_b_fadd(b, tcp, valid_f);
            }

            new_vals[0] = sp;
            new_vals[1] = sqp;
            new_vals[2] = tcp;
            rocke_b_scf_yield(b, new_vals, 3);
        }
        rocke_b_region_leave(b);

        sum_p = outer_for.op->results[0];
        sumsq_p = outer_for.op->results[1];
        tail_count_p = outer_for.op->results[2];

        trip_count = rocke_b_div(b, HW, rocke_b_const_i32(b, BS * VEC));
        main_per_thread_total =
            rocke_b_mul(b, p_n, rocke_b_mul(b, trip_count, rocke_b_const_i32(b, VEC)));
        count_val = rocke_b_fadd(b, rocke_b_sitofp_f32(b, main_per_thread_total), tail_count_p);

        mean_p = rocke_b_fdiv(b, sum_p, count_val);
        m2_p = rocke_b_fsub(b, sumsq_p, rocke_b_fmul(b, mean_p, sum_p));

        if(spec->fast_reduce)
        {
            rocke_welford_block_reduce_stable_wave_prologue(
                b, mean_p, m2_p, count_val, lds_mean, lds_m2, lds_count, tid, BS, spec->wave_size, &mean, &var);
        }
        else
        {
            rocke_welford_block_reduce_stable(
                b, mean_p, m2_p, count_val, lds_mean, lds_m2, lds_count, tid, BS, &mean, &var);
        }

        {
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
            rocke_b_region_enter(b, iff.then_region);
            {
                rocke_value_t* old_mean = rocke_b_global_load_f32(b, RunningMean, channel, 4);
                rocke_value_t* old_var = rocke_b_global_load_f32(b, RunningVar, channel, 4);
                rocke_value_t* one_minus_mom = rocke_b_fsub(b, rocke_b_const_f32(b, 1.0), momentum);
                rocke_value_t* mean_term1 = rocke_b_fmul(b, one_minus_mom, old_mean);
                rocke_value_t* mean_term2 = rocke_b_fmul(b, momentum, mean);
                rocke_value_t* new_running_mean = rocke_b_fadd(b, mean_term1, mean_term2);
                rocke_value_t* var_term1 = rocke_b_fmul(b, one_minus_mom, old_var);
                rocke_value_t* var_term2 = rocke_b_fmul(b, momentum, var);
                rocke_value_t* new_running_var = rocke_b_fadd(b, var_term1, var_term2);
                rocke_b_global_store(b, RunningMean, channel, new_running_mean, 1);
                rocke_b_global_store(b, RunningVar, channel, new_running_var, 1);
                if(spec->save_mean_invstd)
                {
                    rocke_value_t* inv_std_save = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));
                    rocke_b_store_scalar_from_f32(b, Mean, channel, mean, spec->dtype);
                    rocke_b_store_scalar_from_f32(b, InvStd, channel, inv_std_save, spec->dtype);
                }
            }
            rocke_b_region_leave(b);
        }
    }
    else
    {
        mean = rocke_b_global_load_f32(b, RunningMean, channel, 4);
        var = rocke_b_global_load_f32(b, RunningVar, channel, 4);
    }

    inv_std = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));
    gamma_v = rocke_b_load_scalar_as_f32(b, Gamma, channel, spec->dtype);
    beta_v = rocke_b_load_scalar_as_f32(b, Beta, channel, spec->dtype);
    scale = rocke_b_fmul(b, inv_std, gamma_v);

    {
        rocke_for_t outer_for2 = rocke_b_scf_for(b, rocke_b_const_i32(b, 0), p_n, rocke_b_const_i32(b, 1), "n1");
        rocke_b_region_enter(b, outer_for2.body);
        {
            rocke_value_t* n_iv = outer_for2.iv;
            rocke_value_t* row_n = rocke_b_add(b, rocke_b_mul(b, n_iv, p_c), channel);
            rocke_tile_window_t x_tile_n, y_tile_n;
            rocke_value_t *trip_count2, *main_elems2;
            bn_dyn_pass2_ctx_t p2;

            lengths2[0] = 1;
            lengths2[1] = 1;
            origin[0] = row_n;
            origin[1] = rocke_b_const_i32(b, 0);
            rocke_make_tile_window(&x_tile_n, &x_view, lengths2, origin, 2);
            origin[0] = row_n;
            origin[1] = rocke_b_const_i32(b, 0);
            rocke_make_tile_window(&y_tile_n, &y_view, lengths2, origin, 2);

            trip_count2 = rocke_b_div(b, HW, rocke_b_const_i32(b, BS * VEC));
            main_elems2 = rocke_b_mul(b, trip_count2, rocke_b_const_i32(b, BS * VEC));

            p2.mean = mean;
            p2.scale = scale;
            p2.beta_v = beta_v;
            p2.x_tile = &x_tile_n;

            rocke_pass2_row_chunks_dynamic(b,
                                           &y_tile_n,
                                           tid,
                                           BS,
                                           VEC,
                                           main_elems2,
                                           trip_count2,
                                           NULL,
                                           bn_dyn_pass2_body,
                                           &p2,
                                           "hw1");

            for(tc = 0; tc < VEC; ++tc)
            {
                rocke_value_t* idx_col =
                    rocke_b_add(b, rocke_b_add(b, main_elems2, rocke_b_const_i32(b, tc * BS)), tid);
                rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, HW);
                rocke_if_t iff = rocke_b_scf_if(b, valid);
                rocke_b_region_enter(b, iff.then_region);
                {
                    rocke_value_t* flat = rocke_b_add(b, rocke_b_mul(b, row_n, HW), idx_col);
                    rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                    rocke_value_t* y_val = rocke_b_fadd(
                        b, rocke_b_fmul(b, rocke_b_fsub(b, x_raw, mean), scale), beta_v);
                    rocke_b_store_scalar_from_f32(b, Y, flat, y_val, spec->dtype);
                }
                rocke_b_region_leave(b);
            }
        }
        rocke_b_region_leave(b);
    }
    } /* if(NG == 1) */
    else
    {
        /* n_groups > 1: N and H*W split across the CTA's threads
         * simultaneously -- runtime-generic counterpart of
         * build_batchnorm2d's n_groups>1 branch. ``og`` ranges over a
         * RUNTIME N // NG instead of a Python/C-build-time one; each
         * thread's sample index is still n_runtime = og*NG + lidihw, and
         * the per-og row sweep is the same sweep_row_chunks_dynamic +
         * masked-tail idiom the n_groups==1 branch uses, just scoped to
         * hw_threads = BS / NG threads and tid_within_row instead of the
         * full block / tid. */
        int hw_threads = BS / NG;
        rocke_value_t* c_hw_threads = rocke_b_const_i32(b, hw_threads);
        rocke_value_t* lidihw = rocke_b_div(b, tid, c_hw_threads);
        rocke_value_t* tid_within_row = rocke_b_mod(b, tid, c_hw_threads);
        rocke_value_t* n_outer_trip = rocke_b_div(b, p_n, rocke_b_const_i32(b, NG));

        if(spec->training)
        {
            rocke_iter_arg_t outer_iter_args_ng[3];
            rocke_for_t outer_for_ng;
            rocke_value_t *sum_p, *sumsq_p, *tail_count_p;
            rocke_value_t *trip_count_ng, *main_per_thread_total_ng, *count_val, *count_safe;
            rocke_value_t *mean_p, *m2_p;

            outer_iter_args_ng[0].name = "sum_p";
            outer_iter_args_ng[0].init = rocke_b_const_f32(b, 0.0);
            outer_iter_args_ng[1].name = "sumsq_p";
            outer_iter_args_ng[1].init = rocke_b_const_f32(b, 0.0);
            outer_iter_args_ng[2].name = "tail_count_p";
            outer_iter_args_ng[2].init = rocke_b_const_f32(b, 0.0);

            outer_for_ng = rocke_b_scf_for_iter(b,
                                                rocke_b_const_i32(b, 0),
                                                n_outer_trip,
                                                rocke_b_const_i32(b, 1),
                                                outer_iter_args_ng,
                                                3,
                                                "og0",
                                                false,
                                                true);
            rocke_b_region_enter(b, outer_for_ng.body);
            {
                rocke_value_t* og_iv = outer_for_ng.iv;
                rocke_value_t* n_runtime_mul = rocke_b_mul(b, og_iv, rocke_b_const_i32(b, NG));
                rocke_value_t* n_runtime = rocke_b_add(b, n_runtime_mul, lidihw);
                rocke_value_t* row_n = rocke_b_add(b, rocke_b_mul(b, n_runtime, p_c), channel);
                rocke_tile_window_t x_tile_n;
                rocke_iter_arg_t inner_iter_args[2];
                rocke_sweep_dynamic_result_t sweep_res;
                rocke_value_t *sp, *sqp, *tcp;
                rocke_value_t* new_vals[3];

                lengths2[0] = 1;
                lengths2[1] = 1;
                origin[0] = row_n;
                origin[1] = rocke_b_const_i32(b, 0);
                rocke_make_tile_window(&x_tile_n, &x_view, lengths2, origin, 2);

                inner_iter_args[0].name = "sum_p_hw_ng";
                inner_iter_args[0].init = outer_for_ng.iter_vars[0];
                inner_iter_args[1].name = "sumsq_p_hw_ng";
                inner_iter_args[1].init = outer_for_ng.iter_vars[1];

                sweep_res = rocke_sweep_row_chunks_dynamic(b,
                                                           &x_tile_n,
                                                           tid_within_row,
                                                           hw_threads,
                                                           VEC,
                                                           HW,
                                                           inner_iter_args,
                                                           2,
                                                           NULL,
                                                           bn_dyn_pass1_body,
                                                           NULL,
                                                           "hw0_ng");
                sp = sweep_res.final_iter_vals[0];
                sqp = sweep_res.final_iter_vals[1];
                tcp = outer_for_ng.iter_vars[2];

                for(tc = 0; tc < VEC; ++tc)
                {
                    rocke_value_t* idx_col = rocke_b_add(
                        b,
                        rocke_b_add(b, sweep_res.main_elems_per_row, rocke_b_const_i32(b, tc * hw_threads)),
                        tid_within_row);
                    rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, HW);
                    rocke_value_t* idx_safe =
                        rocke_b_smin(b, idx_col, rocke_b_sub(b, HW, rocke_b_const_i32(b, 1)));
                    rocke_value_t* flat = rocke_b_add(b, rocke_b_mul(b, row_n, HW), idx_safe);
                    rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                    rocke_value_t* zero_f = rocke_b_const_f32(b, 0.0);
                    rocke_value_t* x_masked = rocke_b_select(b, valid, x_raw, zero_f);
                    rocke_value_t* one_f2 = rocke_b_const_f32(b, 1.0);
                    rocke_value_t* zero_f2 = rocke_b_const_f32(b, 0.0);
                    rocke_value_t* valid_f = rocke_b_select(b, valid, one_f2, zero_f2);
                    sp = rocke_b_fadd(b, sp, x_masked);
                    sqp = rocke_b_fadd(b, sqp, rocke_b_fmul(b, x_masked, x_masked));
                    tcp = rocke_b_fadd(b, tcp, valid_f);
                }

                new_vals[0] = sp;
                new_vals[1] = sqp;
                new_vals[2] = tcp;
                rocke_b_scf_yield(b, new_vals, 3);
            }
            rocke_b_region_leave(b);

            sum_p = outer_for_ng.op->results[0];
            sumsq_p = outer_for_ng.op->results[1];
            tail_count_p = outer_for_ng.op->results[2];

            trip_count_ng = rocke_b_div(b, HW, rocke_b_const_i32(b, hw_threads * VEC));
            main_per_thread_total_ng =
                rocke_b_mul(b, n_outer_trip, rocke_b_mul(b, trip_count_ng, rocke_b_const_i32(b, VEC)));
            count_val = rocke_b_fadd(b, rocke_b_sitofp_f32(b, main_per_thread_total_ng), tail_count_p);
            count_safe = rocke_b_fmax(b, count_val, rocke_b_const_f32(b, 1.0));

            mean_p = rocke_b_fdiv(b, sum_p, count_safe);
            m2_p = rocke_b_fsub(b, sumsq_p, rocke_b_fmul(b, mean_p, sum_p));

            if(spec->fast_reduce)
            {
                rocke_welford_block_reduce_stable_wave_prologue(
                    b, mean_p, m2_p, count_val, lds_mean, lds_m2, lds_count, tid, BS, spec->wave_size, &mean, &var);
            }
            else
            {
                rocke_welford_block_reduce_stable(
                    b, mean_p, m2_p, count_val, lds_mean, lds_m2, lds_count, tid, BS, &mean, &var);
            }

            {
                rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
                rocke_b_region_enter(b, iff.then_region);
                {
                    rocke_value_t* old_mean = rocke_b_global_load_f32(b, RunningMean, channel, 4);
                    rocke_value_t* old_var = rocke_b_global_load_f32(b, RunningVar, channel, 4);
                    rocke_value_t* one_minus_mom = rocke_b_fsub(b, rocke_b_const_f32(b, 1.0), momentum);
                    rocke_value_t* mean_term1 = rocke_b_fmul(b, one_minus_mom, old_mean);
                    rocke_value_t* mean_term2 = rocke_b_fmul(b, momentum, mean);
                    rocke_value_t* new_running_mean = rocke_b_fadd(b, mean_term1, mean_term2);
                    rocke_value_t* var_term1 = rocke_b_fmul(b, one_minus_mom, old_var);
                    rocke_value_t* var_term2 = rocke_b_fmul(b, momentum, var);
                    rocke_value_t* new_running_var = rocke_b_fadd(b, var_term1, var_term2);
                    rocke_b_global_store(b, RunningMean, channel, new_running_mean, 1);
                    rocke_b_global_store(b, RunningVar, channel, new_running_var, 1);
                    if(spec->save_mean_invstd)
                    {
                        rocke_value_t* inv_std_save = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));
                        rocke_b_store_scalar_from_f32(b, Mean, channel, mean, spec->dtype);
                        rocke_b_store_scalar_from_f32(b, InvStd, channel, inv_std_save, spec->dtype);
                    }
                }
                rocke_b_region_leave(b);
            }
        }
        else
        {
            mean = rocke_b_global_load_f32(b, RunningMean, channel, 4);
            var = rocke_b_global_load_f32(b, RunningVar, channel, 4);
        }

        inv_std = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));
        gamma_v = rocke_b_load_scalar_as_f32(b, Gamma, channel, spec->dtype);
        beta_v = rocke_b_load_scalar_as_f32(b, Beta, channel, spec->dtype);
        scale = rocke_b_fmul(b, inv_std, gamma_v);

        {
            rocke_for_t outer_for2_ng =
                rocke_b_scf_for(b, rocke_b_const_i32(b, 0), n_outer_trip, rocke_b_const_i32(b, 1), "og1");
            rocke_b_region_enter(b, outer_for2_ng.body);
            {
                rocke_value_t* og_iv = outer_for2_ng.iv;
                rocke_value_t* n_runtime_mul = rocke_b_mul(b, og_iv, rocke_b_const_i32(b, NG));
                rocke_value_t* n_runtime = rocke_b_add(b, n_runtime_mul, lidihw);
                rocke_value_t* row_n = rocke_b_add(b, rocke_b_mul(b, n_runtime, p_c), channel);
                rocke_tile_window_t x_tile_n, y_tile_n;
                rocke_value_t *trip_count2_ng, *main_elems2_ng;
                bn_dyn_pass2_ctx_t p2;

                lengths2[0] = 1;
                lengths2[1] = 1;
                origin[0] = row_n;
                origin[1] = rocke_b_const_i32(b, 0);
                rocke_make_tile_window(&x_tile_n, &x_view, lengths2, origin, 2);
                origin[0] = row_n;
                origin[1] = rocke_b_const_i32(b, 0);
                rocke_make_tile_window(&y_tile_n, &y_view, lengths2, origin, 2);

                trip_count2_ng = rocke_b_div(b, HW, rocke_b_const_i32(b, hw_threads * VEC));
                main_elems2_ng = rocke_b_mul(b, trip_count2_ng, rocke_b_const_i32(b, hw_threads * VEC));

                p2.mean = mean;
                p2.scale = scale;
                p2.beta_v = beta_v;
                p2.x_tile = &x_tile_n;

                rocke_pass2_row_chunks_dynamic(b,
                                               &y_tile_n,
                                               tid_within_row,
                                               hw_threads,
                                               VEC,
                                               main_elems2_ng,
                                               trip_count2_ng,
                                               NULL,
                                               bn_dyn_pass2_body,
                                               &p2,
                                               "hw1_ng");

                for(tc = 0; tc < VEC; ++tc)
                {
                    rocke_value_t* idx_col = rocke_b_add(
                        b,
                        rocke_b_add(b, main_elems2_ng, rocke_b_const_i32(b, tc * hw_threads)),
                        tid_within_row);
                    rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, HW);
                    rocke_if_t iff = rocke_b_scf_if(b, valid);
                    rocke_b_region_enter(b, iff.then_region);
                    {
                        rocke_value_t* flat = rocke_b_add(b, rocke_b_mul(b, row_n, HW), idx_col);
                        rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                        rocke_value_t* y_val = rocke_b_fadd(
                            b, rocke_b_fmul(b, rocke_b_fsub(b, x_raw, mean), scale), beta_v);
                        rocke_b_store_scalar_from_f32(b, Y, flat, y_val, spec->dtype);
                    }
                    rocke_b_region_leave(b);
                }
            }
            rocke_b_region_leave(b);
        }
    } /* else NG > 1 */
    } /* for(cb = 0; cb < CPB; ++cb) */

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

/* ------------------------------------------------------------------ *
 * build_batchnorm2d
 * ------------------------------------------------------------------ */

rocke_kernel_def_t* rocke_build_batchnorm2d(rocke_ir_builder_t* b,
                                            const rocke_batchnorm2d_spec_t* spec)
{
    const rocke_type_t* io_ty;
    int BS;
    int VEC;
    int N;
    int C;
    int HW;
    int ept_row;
    int main_elems_per_row;
    int tail_chunks;
    bool two_pass;
    bool ok;

    rocke_value_t* X;
    rocke_value_t* Gamma;
    rocke_value_t* Beta;
    rocke_value_t* Y;
    rocke_value_t* RunningMean;
    rocke_value_t* RunningVar;
    rocke_value_t* Mean = NULL;
    rocke_value_t* InvStd = NULL;
    rocke_value_t* eps;
    rocke_value_t* momentum;

    rocke_value_t* tid;
    rocke_value_t* channel;

    rocke_tensor_view_t x_view;
    rocke_tensor_view_t y_view;

    rocke_tensor_view_t lds_mean_v;
    rocke_tensor_view_t lds_m2_v;
    rocke_tensor_view_t lds_count_v;
    rocke_value_t* lds_mean = NULL;
    rocke_value_t* lds_m2 = NULL;
    rocke_value_t* lds_count = NULL;

    rocke_value_t* mean = NULL;
    rocke_value_t* var = NULL;
    rocke_value_t* inv_std;
    rocke_value_t* gamma_v;
    rocke_value_t* beta_v;
    rocke_value_t* scale;

    int shape2[2];
    int lengths2[2];
    rocke_value_t* origin[2];
    rocke_param_opts_t opts;

    rocke_value_t** row_values = NULL;
    rocke_tile_window_t* x_tiles = NULL;
    rocke_value_t*** cached_by_n = NULL;
    int* num_cached_by_n = NULL;

    int n;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }

    ok = rocke_batchnorm2d_is_valid_spec(spec, "gfx950", NULL, 0);
    if(!ok)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "invalid batchnorm2d spec");
        return NULL;
    }

    if(spec->dynamic)
    {
        return rocke_build_batchnorm2d_dynamic(b, spec);
    }

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    BS = spec->block_size;
    VEC = spec->vec;
    N = spec->n;
    C = spec->c;
    HW = rocke_batchnorm2d_hw(spec);
    /* Mirrors the Python builder's ``chunks_main``/``tail_chunks`` block: NG==1's
     * row sweep now tolerates an HW not divisible by block_size*vec via a
     * scalar tail (see the ``tail_chunks`` loops below), so ``ept_row`` here is
     * the vectorized-main-loop count only -- identical to the historical value
     * when HW % (BS*VEC) == 0, strictly smaller otherwise. */
    {
        int vec_bs = BS * VEC;
        int chunks_main = HW / vec_bs;
        int tail_per_row;
        ept_row = chunks_main * VEC;
        main_elems_per_row = ept_row * BS;
        tail_per_row = HW - main_elems_per_row;
        tail_chunks = tail_per_row ? (tail_per_row + BS - 1) / BS : 0;
    }
    two_pass = rocke_row_norm_needs_two_pass(rocke_batchnorm2d_elems_per_thread(spec),
                                             ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* --- params (ABI order matches the Python build) --- */
    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    X = rocke_b_param(b, "X", rocke_ptr_type(b, io_ty, "global"), &opts);
    Gamma = rocke_b_param(b, "Gamma", rocke_ptr_type(b, io_ty, "global"), &opts);
    Beta = rocke_b_param(b, "Beta", rocke_ptr_type(b, io_ty, "global"), &opts);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    Y = rocke_b_param(b, "Y", rocke_ptr_type(b, io_ty, "global"), &opts);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    RunningMean = rocke_b_param(b, "RunningMean", rocke_ptr_type(b, rocke_f32(), "global"), &opts);
    RunningVar = rocke_b_param(b, "RunningVar", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    if(spec->save_mean_invstd)
    {
        memset(&opts, 0, sizeof opts);
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        Mean = rocke_b_param(b, "Mean", rocke_ptr_type(b, io_ty, "global"), &opts);
        InvStd = rocke_b_param(b, "InvStd", rocke_ptr_type(b, io_ty, "global"), &opts);
    }

    (void)rocke_b_param(b, "N", rocke_i32(), NULL);
    (void)rocke_b_param(b, "C", rocke_i32(), NULL);
    (void)rocke_b_param(b, "H", rocke_i32(), NULL);
    (void)rocke_b_param(b, "W", rocke_i32(), NULL);
    eps = rocke_b_param(b, "eps", rocke_f32(), NULL);
    momentum = rocke_b_param(b, "momentum", rocke_f32(), NULL);

    tid = rocke_b_thread_id_x(b);
    {
        rocke_value_t* base_channel_early = rocke_b_block_id_x(b);
        channel = base_channel_early; /* reused as the loop's base_channel below */
    }

    /* Packed (N*C, H*W) view: see the Python module docstring for the NCHW
     * row-per-(n,c) layout rationale. */
    shape2[0] = N * C;
    shape2[1] = HW;
    rocke_make_naive_tensor_view_packed(&x_view, X, shape2, 2, io_ty);
    rocke_make_naive_tensor_view_packed(&y_view, Y, shape2, 2, io_ty);

    if(spec->training)
    {
        int ldsshape[1];
        ldsshape[0] = BS;
        rocke_make_lds_view(b, &lds_mean_v, rocke_f32(), ldsshape, 1, "lds_mean", NULL);
        rocke_make_lds_view(b, &lds_m2_v, rocke_f32(), ldsshape, 1, "lds_m2", NULL);
        rocke_make_lds_view(b, &lds_count_v, rocke_f32(), ldsshape, 1, "lds_count", NULL);
        lds_mean = lds_mean_v.base;
        lds_m2 = lds_m2_v.base;
        lds_count = lds_count_v.base;
    }

    row_values = (rocke_value_t**)malloc(sizeof(rocke_value_t*) * (size_t)N);
    x_tiles = (rocke_tile_window_t*)malloc(sizeof(rocke_tile_window_t) * (size_t)N);
    cached_by_n = (rocke_value_t***)calloc((size_t)N, sizeof(rocke_value_t**));
    num_cached_by_n = (int*)calloc((size_t)N, sizeof(int));

    {
        /* channels_per_block (default 1, byte-identical to the original
         * single-channel-per-CTA emission): a build-time loop over the CPB
         * channels this CTA owns, run sequentially with the full block of
         * threads on each in turn. See the Python builder's matching
         * comment for the LDS-reuse-barrier rationale. */
        rocke_value_t* base_channel = channel; /* captured right after tid, above */
        int CPB = spec->channels_per_block;
        int NG = spec->n_groups;
        int cb;

        for(cb = 0; cb < CPB; ++cb)
        {
            bn_pass1_ctx_t p1;

            if(CPB == 1)
            {
                channel = base_channel;
            }
            else
            {
                /* Match Python's left-to-right argument evaluation
                 * (b.mul(...) emitted before b.const_i32(cb)): C/C++
                 * argument evaluation order is unspecified, so each operand
                 * is materialised into its own statement in emission order
                 * rather than nested in the rocke_b_add call. */
                rocke_value_t* scaled = rocke_b_mul(b, base_channel, rocke_b_const_i32(b, CPB));
                rocke_value_t* cb_const = rocke_b_const_i32(b, cb);
                channel = rocke_b_add(b, scaled, cb_const);
                if(spec->training && cb > 0)
                {
                    rocke_b_sync(b);
                }
            }

            lengths2[0] = 1;
            lengths2[1] = HW;

            if(NG == 1)
            {
            p1.sum_p = rocke_b_const_f32(b, 0.0);
            p1.sumsq_p = rocke_b_const_f32(b, 0.0);
            rocke_value_t* tail_count_p = tail_chunks ? rocke_b_const_f32(b, 0.0) : NULL;

            for(n = 0; n < N; ++n)
            {
                rocke_value_t* row_n = rocke_b_add(b, rocke_b_const_i32(b, n * C), channel);
                row_values[n] = row_n;
                origin[0] = row_n;
                origin[1] = rocke_b_const_i32(b, 0);
                rocke_make_tile_window(&x_tiles[n], &x_view, lengths2, origin, 2);

                if(spec->training)
                {
                    rocke_row_chunk_sweep_result_t res =
                        rocke_sweep_row_chunks(b,
                                               &x_tiles[n],
                                               tid,
                                               BS,
                                               VEC,
                                               ept_row,
                                               NULL, /* row -- origin already placed */
                                               bn_pass1_body,
                                               &p1,
                                               !two_pass);
                    cached_by_n[n] = res.cached;
                    num_cached_by_n[n] = res.num_cached;

                    /* Scalar tail, branch-free (masked via rocke_b_select,
                     * not an scf_if guard): mirrors the Python builder's
                     * tail_chunks block exactly -- the load index is
                     * clamped with rocke_b_smin to stay inside this row's
                     * HW span before the mask is applied to its value, so
                     * an out-of-range lane still issues an in-bounds load,
                     * just a discarded one. */
                    {
                        int tc;
                        for(tc = 0; tc < tail_chunks; ++tc)
                        {
                            rocke_value_t* idx_col = rocke_b_add(
                                b, rocke_b_const_i32(b, main_elems_per_row + tc * BS), tid);
                            rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, rocke_b_const_i32(b, HW));
                            rocke_value_t* idx_safe =
                                rocke_b_smin(b, idx_col, rocke_b_const_i32(b, HW - 1));
                            rocke_value_t* flat =
                                rocke_b_add(b, rocke_b_mul(b, row_n, rocke_b_const_i32(b, HW)), idx_safe);
                            rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                            rocke_value_t* x_masked = rocke_b_select(b, valid, x_raw, rocke_b_const_f32(b, 0.0));
                            rocke_value_t* valid_f =
                                rocke_b_select(b, valid, rocke_b_const_f32(b, 1.0), rocke_b_const_f32(b, 0.0));
                            p1.sum_p = rocke_b_fadd(b, p1.sum_p, x_masked);
                            p1.sumsq_p = rocke_b_fadd(b, p1.sumsq_p, rocke_b_fmul(b, x_masked, x_masked));
                            tail_count_p = rocke_b_fadd(b, tail_count_p, valid_f);
                        }
                    }
                }
            }

            if(spec->training)
            {
                rocke_value_t* mean_p;
                rocke_value_t* m2_p;
                rocke_value_t* count_val = NULL;
                double count_p = 0.0;

                if(tail_chunks)
                {
                    count_val = rocke_b_fadd(
                        b, rocke_b_const_f32(b, (double)(N * ept_row)), tail_count_p);
                    mean_p = rocke_b_fdiv(b, p1.sum_p, count_val);
                }
                else
                {
                    rocke_value_t* inv_count_p;
                    count_p = (double)rocke_batchnorm2d_elems_per_thread(spec);
                    inv_count_p = rocke_b_const_f32(b, 1.0 / count_p);
                    mean_p = rocke_b_fmul(b, p1.sum_p, inv_count_p);
                }
                m2_p = rocke_b_fsub(b, p1.sumsq_p, rocke_b_fmul(b, mean_p, p1.sum_p));
                if(!tail_chunks)
                {
                    /* Byte-identical to the pre-tail-support emission: two
                     * independent rocke_b_const_f32(count_p) calls, not a
                     * single value reused, and emitted in the same relative
                     * position (after m2_p) -- deduplicating or reordering
                     * either would shift the op/value numbering the Python
                     * engine must match byte-for-byte. */
                    count_val = rocke_b_const_f32(b, count_p);
                }

                if(spec->fast_reduce)
                {
                    rocke_welford_block_reduce_stable_wave_prologue(b,
                                                                    mean_p,
                                                                    m2_p,
                                                                    count_val,
                                                                    lds_mean,
                                                                    lds_m2,
                                                                    lds_count,
                                                                    tid,
                                                                    BS,
                                                                    spec->wave_size,
                                                                    &mean,
                                                                    &var);
                }
                else
                {
                    rocke_welford_block_reduce_stable(b,
                                                      mean_p,
                                                      m2_p,
                                                      count_val,
                                                      lds_mean,
                                                      lds_m2,
                                                      lds_count,
                                                      tid,
                                                      BS,
                                                      &mean,
                                                      &var);
                }

                {
                    rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
                    rocke_value_t* old_mean;
                    rocke_value_t* old_var;
                    rocke_value_t* one_minus_mom;
                    rocke_value_t* new_running_mean;
                    rocke_value_t* new_running_var;

                    rocke_b_region_enter(b, iff.then_region);
                    old_mean = rocke_b_global_load_f32(b, RunningMean, channel, 4);
                    old_var = rocke_b_global_load_f32(b, RunningVar, channel, 4);
                    one_minus_mom = rocke_b_fsub(b, rocke_b_const_f32(b, 1.0), momentum);
                    {
                        /* Match Python's left-to-right argument evaluation
                         * (b.fmul(one_minus_mom, old_mean) emitted before
                         * b.fmul(momentum, mean)): C/C++ argument evaluation
                         * order is unspecified, so each operand MUST be
                         * materialised into its own statement in emission
                         * order rather than nested in the fadd call. */
                        rocke_value_t* term_a = rocke_b_fmul(b, one_minus_mom, old_mean);
                        rocke_value_t* term_b = rocke_b_fmul(b, momentum, mean);
                        new_running_mean = rocke_b_fadd(b, term_a, term_b);
                    }
                    {
                        rocke_value_t* term_a = rocke_b_fmul(b, one_minus_mom, old_var);
                        rocke_value_t* term_b = rocke_b_fmul(b, momentum, var);
                        new_running_var = rocke_b_fadd(b, term_a, term_b);
                    }
                    rocke_b_global_store(b, RunningMean, channel, new_running_mean, 1);
                    rocke_b_global_store(b, RunningVar, channel, new_running_var, 1);
                    if(spec->save_mean_invstd)
                    {
                        rocke_value_t* inv_std_save = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));
                        rocke_b_store_scalar_from_f32(b, Mean, channel, mean, spec->dtype);
                        rocke_b_store_scalar_from_f32(b, InvStd, channel, inv_std_save, spec->dtype);
                    }
                    rocke_b_region_leave(b);
                }
            }
            else
            {
                mean = rocke_b_global_load_f32(b, RunningMean, channel, 4);
                var = rocke_b_global_load_f32(b, RunningVar, channel, 4);
            }

            inv_std = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));

            /* Gamma / Beta are per-channel scalars -- load once per CTA. */
            gamma_v = rocke_b_load_scalar_as_f32(b, Gamma, channel, spec->dtype);
            beta_v = rocke_b_load_scalar_as_f32(b, Beta, channel, spec->dtype);
            scale = rocke_b_fmul(b, inv_std, gamma_v);

            /* --- Pass 2: N-loop of per-sample row writes --- */
            for(n = 0; n < N; ++n)
            {
                rocke_tile_window_t y_tile_n;
                bn_pass2_ctx_t p2;

                origin[0] = row_values[n];
                origin[1] = rocke_b_const_i32(b, 0);
                rocke_make_tile_window(&y_tile_n, &y_view, lengths2, origin, 2);

                p2.two_pass = two_pass;
                p2.training = spec->training;
                p2.vec = VEC;
                p2.x_tile_n = &x_tiles[n];
                p2.mean = mean;
                p2.scale = scale;
                p2.beta_v = beta_v;

                rocke_pass2_row_chunks(b,
                                       &y_tile_n,
                                       tid,
                                       BS,
                                       VEC,
                                       ept_row,
                                       NULL, /* row -- origin already placed */
                                       bn_pass2_body,
                                       &p2,
                                       cached_by_n[n],
                                       num_cached_by_n[n]);

                /* Scalar tail write-back. Unlike pass 1's masked reduction,
                 * a store cannot be branch-free (clamping an out-of-range
                 * lane's index would make multiple lanes race to store into
                 * the SAME element), so this uses a real scf_if guard,
                 * matching the Python builder's elementwise.py-derived
                 * idiom, and always reloads X directly (the tail is at most
                 * BS*VEC - 1 elements -- not worth caching). */
                {
                    int tc;
                    for(tc = 0; tc < tail_chunks; ++tc)
                    {
                        rocke_value_t* idx_col = rocke_b_add(
                            b, rocke_b_const_i32(b, main_elems_per_row + tc * BS), tid);
                        rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, rocke_b_const_i32(b, HW));
                        rocke_if_t tail_iff = rocke_b_scf_if(b, valid);
                        rocke_value_t* flat;
                        rocke_value_t* x_raw;
                        rocke_value_t* y_val;

                        rocke_b_region_enter(b, tail_iff.then_region);
                        flat = rocke_b_add(b, rocke_b_mul(b, row_values[n], rocke_b_const_i32(b, HW)), idx_col);
                        x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                        y_val = rocke_b_fadd(b, rocke_b_fmul(b, rocke_b_fsub(b, x_raw, mean), scale), beta_v);
                        rocke_b_store_scalar_from_f32(b, Y, flat, y_val, spec->dtype);
                        rocke_b_region_leave(b);
                    }
                }
            }
            }
            else
            {
                /* n_groups > 1: N and H*W split across the CTA's threads
                 * simultaneously (mirrors MIOpen's spatial batchnorm thread
                 * mapping). Training and inference are kept as two separate
                 * emission orders below (not a shared preamble) so the
                 * pre-existing inference emission stays byte-for-byte
                 * identical to before training support was added -- only
                 * the spec->training arm is new. */
                int hw_threads = BS / NG;
                /* Same tail shape as NG==1's chunks_main/tail_chunks (see
                 * that block's comment above), scoped to hw_threads instead
                 * of BS: the vectorized main sweep covers whole
                 * hw_threads*vec chunks of one og's row, and the remaining
                 * tail_per_row_ng elements are handled by a masked scalar
                 * tail below. When HW % (hw_threads*vec) == 0 (the old exact-
                 * divisibility case), tail_chunks_ng == 0 and every path
                 * below collapses back to the original emission. */
                int ept_row_ng;
                int main_elems_per_row_ng;
                int tail_chunks_ng;
                {
                    int chunks_main_ng = HW / (hw_threads * VEC);
                    int tail_per_row_ng;
                    ept_row_ng = chunks_main_ng * VEC;
                    main_elems_per_row_ng = ept_row_ng * hw_threads;
                    tail_per_row_ng = HW - main_elems_per_row_ng;
                    tail_chunks_ng =
                        tail_per_row_ng ? (tail_per_row_ng + hw_threads - 1) / hw_threads : 0;
                }

                if(spec->training)
                {
                    /* Each thread accumulates its own Welford partial over
                     * its assigned (og, tid_within_row) slice of elements,
                     * then merges via the SAME welford_block_reduce_stable/
                     * _wave_prologue call the NG==1 path uses over the full
                     * tid/block_size=BS -- that merge only needs each of the
                     * BS threads to hand in its own (mean, M2, count)
                     * triple, it does not care how a thread's slice was
                     * chosen, so no new reduction helper or two-tier merge
                     * is needed. Unlike NG==1's training path this never
                     * caches X in registers across pass 1/pass 2 (cache=0
                     * below) -- see the Python builder's matching comment
                     * for why that scope cut doesn't cost much for the
                     * small-HW shapes this lever targets. */
                    rocke_value_t* c_hw_threads = rocke_b_const_i32(b, hw_threads);
                    rocke_value_t* lidihw = rocke_b_div(b, tid, c_hw_threads);
                    rocke_value_t* tid_within_row = rocke_b_mod(b, tid, c_hw_threads);
                    int num_og = N / NG;
                    rocke_value_t** og_rows =
                        (rocke_value_t**)malloc(sizeof(rocke_value_t*) * (size_t)num_og);
                    rocke_tile_window_t* og_x_tiles =
                        (rocke_tile_window_t*)malloc(sizeof(rocke_tile_window_t) * (size_t)num_og);
                    bn_pass1_ctx_t p1ng;
                    rocke_value_t* mean_p;
                    rocke_value_t* m2_p;
                    rocke_value_t* count_val;
                    rocke_value_t* inv_count_p;
                    double count_p;
                    int og;

                    for(og = 0; og < num_og; ++og)
                    {
                        rocke_value_t* n_runtime =
                            rocke_b_add(b, rocke_b_const_i32(b, og * NG), lidihw);
                        rocke_value_t* row_n =
                            rocke_b_add(b, rocke_b_mul(b, n_runtime, rocke_b_const_i32(b, C)), channel);
                        og_rows[og] = row_n;
                        origin[0] = row_n;
                        origin[1] = rocke_b_const_i32(b, 0);
                        rocke_make_tile_window(&og_x_tiles[og], &x_view, lengths2, origin, 2);
                    }

                    rocke_value_t* tail_count_p_ng = tail_chunks_ng ? rocke_b_const_f32(b, 0.0) : NULL;

                    p1ng.sum_p = rocke_b_const_f32(b, 0.0);
                    p1ng.sumsq_p = rocke_b_const_f32(b, 0.0);

                    for(og = 0; og < num_og; ++og)
                    {
                        rocke_sweep_row_chunks(b,
                                               &og_x_tiles[og],
                                               tid_within_row,
                                               hw_threads,
                                               VEC,
                                               ept_row_ng,
                                               NULL, /* row -- origin already placed */
                                               bn_pass1_body,
                                               &p1ng,
                                               0 /* cache = false */);

                        /* Masked scalar tail, same idiom as NG==1's pass-1
                         * tail (clamp + select, not scf_if, since a
                         * reduction contribution is naturally zero-able). */
                        {
                            int tc;
                            for(tc = 0; tc < tail_chunks_ng; ++tc)
                            {
                                rocke_value_t* idx_col = rocke_b_add(
                                    b,
                                    rocke_b_const_i32(b, main_elems_per_row_ng + tc * hw_threads),
                                    tid_within_row);
                                rocke_value_t* valid =
                                    rocke_b_cmp_lt(b, idx_col, rocke_b_const_i32(b, HW));
                                rocke_value_t* idx_safe =
                                    rocke_b_smin(b, idx_col, rocke_b_const_i32(b, HW - 1));
                                rocke_value_t* flat = rocke_b_add(
                                    b, rocke_b_mul(b, og_rows[og], rocke_b_const_i32(b, HW)), idx_safe);
                                rocke_value_t* x_raw =
                                    rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                                rocke_value_t* x_masked =
                                    rocke_b_select(b, valid, x_raw, rocke_b_const_f32(b, 0.0));
                                rocke_value_t* valid_f = rocke_b_select(
                                    b, valid, rocke_b_const_f32(b, 1.0), rocke_b_const_f32(b, 0.0));
                                p1ng.sum_p = rocke_b_fadd(b, p1ng.sum_p, x_masked);
                                p1ng.sumsq_p =
                                    rocke_b_fadd(b, p1ng.sumsq_p, rocke_b_fmul(b, x_masked, x_masked));
                                tail_count_p_ng = rocke_b_fadd(b, tail_count_p_ng, valid_f);
                            }
                        }
                    }

                    if(tail_chunks_ng)
                    {
                        rocke_value_t* count_safe;
                        count_val = rocke_b_fadd(
                            b,
                            rocke_b_const_f32(b, (double)(num_og * ept_row_ng)),
                            tail_count_p_ng);
                        /* A thread whose tid_within_row falls entirely
                         * outside every og's valid HW range gets a real
                         * count_val of 0 (sum_p is 0 too) -- dividing by
                         * max(count_val, 1) instead of count_val avoids a
                         * literal 0/0 = NaN that would otherwise poison the
                         * block merge below even at weight 0 (its delta
                         * term is NaN * 0 = NaN, not 0). The merge still
                         * uses the REAL count_val as that thread's weight. */
                        count_safe = rocke_b_fmax(b, count_val, rocke_b_const_f32(b, 1.0));
                        mean_p = rocke_b_fdiv(b, p1ng.sum_p, count_safe);
                        m2_p = rocke_b_fsub(b, p1ng.sumsq_p, rocke_b_fmul(b, mean_p, p1ng.sum_p));
                    }
                    else
                    {
                        /* No-tail formula: byte-identical to the original
                         * (pre-tail-support) emission -- a single
                         * count_val = rocke_b_const_f32(count_p) call
                         * emitted right after m2_p and reused by both
                         * Welford branches below. */
                        count_p = (double)rocke_batchnorm2d_elems_per_thread(spec);
                        inv_count_p = rocke_b_const_f32(b, 1.0 / count_p);
                        mean_p = rocke_b_fmul(b, p1ng.sum_p, inv_count_p);
                        m2_p = rocke_b_fsub(b, p1ng.sumsq_p, rocke_b_fmul(b, mean_p, p1ng.sum_p));
                        count_val = rocke_b_const_f32(b, count_p);
                    }

                    if(spec->fast_reduce)
                    {
                        rocke_welford_block_reduce_stable_wave_prologue(b,
                                                                        mean_p,
                                                                        m2_p,
                                                                        count_val,
                                                                        lds_mean,
                                                                        lds_m2,
                                                                        lds_count,
                                                                        tid,
                                                                        BS,
                                                                        spec->wave_size,
                                                                        &mean,
                                                                        &var);
                    }
                    else
                    {
                        rocke_welford_block_reduce_stable(b,
                                                          mean_p,
                                                          m2_p,
                                                          count_val,
                                                          lds_mean,
                                                          lds_m2,
                                                          lds_count,
                                                          tid,
                                                          BS,
                                                          &mean,
                                                          &var);
                    }

                    {
                        rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
                        rocke_value_t* old_mean;
                        rocke_value_t* old_var;
                        rocke_value_t* one_minus_mom;
                        rocke_value_t* new_running_mean;
                        rocke_value_t* new_running_var;

                        rocke_b_region_enter(b, iff.then_region);
                        old_mean = rocke_b_global_load_f32(b, RunningMean, channel, 4);
                        old_var = rocke_b_global_load_f32(b, RunningVar, channel, 4);
                        one_minus_mom = rocke_b_fsub(b, rocke_b_const_f32(b, 1.0), momentum);
                        {
                            rocke_value_t* term_a = rocke_b_fmul(b, one_minus_mom, old_mean);
                            rocke_value_t* term_b = rocke_b_fmul(b, momentum, mean);
                            new_running_mean = rocke_b_fadd(b, term_a, term_b);
                        }
                        {
                            rocke_value_t* term_a = rocke_b_fmul(b, one_minus_mom, old_var);
                            rocke_value_t* term_b = rocke_b_fmul(b, momentum, var);
                            new_running_var = rocke_b_fadd(b, term_a, term_b);
                        }
                        rocke_b_global_store(b, RunningMean, channel, new_running_mean, 1);
                        rocke_b_global_store(b, RunningVar, channel, new_running_var, 1);
                        if(spec->save_mean_invstd)
                        {
                            rocke_value_t* inv_std_save = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));
                            rocke_b_store_scalar_from_f32(b, Mean, channel, mean, spec->dtype);
                            rocke_b_store_scalar_from_f32(b, InvStd, channel, inv_std_save, spec->dtype);
                        }
                        rocke_b_region_leave(b);
                    }

                    inv_std = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));
                    gamma_v = rocke_b_load_scalar_as_f32(b, Gamma, channel, spec->dtype);
                    beta_v = rocke_b_load_scalar_as_f32(b, Beta, channel, spec->dtype);
                    scale = rocke_b_fmul(b, inv_std, gamma_v);

                    for(og = 0; og < num_og; ++og)
                    {
                        rocke_tile_window_t y_tile_n_ng;
                        bn_pass2_ctx_t p2;

                        origin[0] = og_rows[og];
                        origin[1] = rocke_b_const_i32(b, 0);
                        rocke_make_tile_window(&y_tile_n_ng, &y_view, lengths2, origin, 2);

                        p2.two_pass = true; /* always reload: no register cache */
                        p2.training = true;
                        p2.vec = VEC;
                        p2.x_tile_n = &og_x_tiles[og];
                        p2.mean = mean;
                        p2.scale = scale;
                        p2.beta_v = beta_v;

                        rocke_pass2_row_chunks(b,
                                               &y_tile_n_ng,
                                               tid_within_row,
                                               hw_threads,
                                               VEC,
                                               ept_row_ng,
                                               NULL, /* row -- origin already placed */
                                               bn_pass2_body,
                                               &p2,
                                               NULL,
                                               0);

                        /* Masked scalar tail write-back, same idiom as
                         * NG==1's pass-2 tail: a real scf_if guard, not
                         * select -- a store cannot be branch-free. */
                        {
                            int tc;
                            for(tc = 0; tc < tail_chunks_ng; ++tc)
                            {
                                rocke_value_t* idx_col = rocke_b_add(
                                    b,
                                    rocke_b_const_i32(b, main_elems_per_row_ng + tc * hw_threads),
                                    tid_within_row);
                                rocke_value_t* valid =
                                    rocke_b_cmp_lt(b, idx_col, rocke_b_const_i32(b, HW));
                                rocke_if_t tail_iff = rocke_b_scf_if(b, valid);
                                rocke_value_t* flat;
                                rocke_value_t* x_raw;
                                rocke_value_t* y_val;

                                rocke_b_region_enter(b, tail_iff.then_region);
                                flat = rocke_b_add(
                                    b, rocke_b_mul(b, og_rows[og], rocke_b_const_i32(b, HW)), idx_col);
                                x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                                y_val = rocke_b_fadd(
                                    b, rocke_b_fmul(b, rocke_b_fsub(b, x_raw, mean), scale), beta_v);
                                rocke_b_store_scalar_from_f32(b, Y, flat, y_val, spec->dtype);
                                rocke_b_region_leave(b);
                            }
                        }
                    }

                    free(og_rows);
                    free(og_x_tiles);
                }
                else
                {
                    /* Inference: emission order preserved byte-for-byte
                     * from before training support was added above. */
                    rocke_value_t* c_hw_threads;
                    rocke_value_t* lidihw;
                    rocke_value_t* tid_within_row;
                    int og;

                    mean = rocke_b_global_load_f32(b, RunningMean, channel, 4);
                    var = rocke_b_global_load_f32(b, RunningVar, channel, 4);
                    inv_std = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));
                    gamma_v = rocke_b_load_scalar_as_f32(b, Gamma, channel, spec->dtype);
                    beta_v = rocke_b_load_scalar_as_f32(b, Beta, channel, spec->dtype);
                    scale = rocke_b_fmul(b, inv_std, gamma_v);

                    c_hw_threads = rocke_b_const_i32(b, hw_threads);
                    lidihw = rocke_b_div(b, tid, c_hw_threads);
                    tid_within_row = rocke_b_mod(b, tid, c_hw_threads);

                    for(og = 0; og < N / NG; ++og)
                    {
                        rocke_value_t* n_runtime =
                            rocke_b_add(b, rocke_b_const_i32(b, og * NG), lidihw);
                        rocke_value_t* row_n =
                            rocke_b_add(b, rocke_b_mul(b, n_runtime, rocke_b_const_i32(b, C)), channel);
                        rocke_tile_window_t y_tile_n_ng;
                        rocke_tile_window_t x_tile_n_ng;
                        bn_pass2_ctx_t p2;

                        origin[0] = row_n;
                        origin[1] = rocke_b_const_i32(b, 0);
                        rocke_make_tile_window(&y_tile_n_ng, &y_view, lengths2, origin, 2);
                        origin[0] = row_n;
                        origin[1] = rocke_b_const_i32(b, 0);
                        rocke_make_tile_window(&x_tile_n_ng, &x_view, lengths2, origin, 2);

                        p2.two_pass = false;
                        p2.training = false;
                        p2.vec = VEC;
                        p2.x_tile_n = &x_tile_n_ng;
                        p2.mean = mean;
                        p2.scale = scale;
                        p2.beta_v = beta_v;

                        rocke_pass2_row_chunks(b,
                                               &y_tile_n_ng,
                                               tid_within_row,
                                               hw_threads,
                                               VEC,
                                               ept_row_ng,
                                               NULL, /* row -- origin already placed */
                                               bn_pass2_body,
                                               &p2,
                                               NULL,
                                               0);

                        /* Masked scalar tail write-back -- same idiom as
                         * the training sub-branch's pass-2 tail above. */
                        {
                            int tc;
                            for(tc = 0; tc < tail_chunks_ng; ++tc)
                            {
                                rocke_value_t* idx_col = rocke_b_add(
                                    b,
                                    rocke_b_const_i32(b, main_elems_per_row_ng + tc * hw_threads),
                                    tid_within_row);
                                rocke_value_t* valid =
                                    rocke_b_cmp_lt(b, idx_col, rocke_b_const_i32(b, HW));
                                rocke_if_t tail_iff = rocke_b_scf_if(b, valid);
                                rocke_value_t* flat;
                                rocke_value_t* x_raw;
                                rocke_value_t* y_val;

                                rocke_b_region_enter(b, tail_iff.then_region);
                                flat = rocke_b_add(
                                    b, rocke_b_mul(b, row_n, rocke_b_const_i32(b, HW)), idx_col);
                                x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                                y_val = rocke_b_fadd(
                                    b, rocke_b_fmul(b, rocke_b_fsub(b, x_raw, mean), scale), beta_v);
                                rocke_b_store_scalar_from_f32(b, Y, flat, y_val, spec->dtype);
                                rocke_b_region_leave(b);
                            }
                        }
                    }
                }
            }
        }
    }

    free(row_values);
    free(x_tiles);
    free(cached_by_n);
    free(num_cached_by_n);

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_batchnorm2d_new(rocke_ir_builder_t* b,
                                                const rocke_batchnorm2d_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_batchnorm2d_kernel_name(spec, name, sizeof name) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_batchnorm2d(b, spec);
    });
}

/* ------------------------------------------------------------------ *
 * grid / signature
 * ------------------------------------------------------------------ */

rocke_status_t
    rocke_batchnorm2d_grid(int c, const rocke_batchnorm2d_spec_t* spec, int out[3])
{
    if(out == NULL || spec == NULL || spec->channels_per_block <= 0)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = c / spec->channels_per_block;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_batchnorm2d_signature(rocke_signature_builder_t* sb,
                                           const rocke_batchnorm2d_spec_t* spec,
                                           const rocke_sig_entry_t** out_items,
                                           size_t* out_count)
{
    if(sb == NULL || spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    rocke_signature_builder_ptr(sb, "X", spec->dtype, NULL);
    rocke_signature_builder_ptr(sb, "Gamma", spec->dtype, NULL);
    rocke_signature_builder_ptr(sb, "Beta", spec->dtype, NULL);
    rocke_signature_builder_ptr(sb, "Y", spec->dtype, NULL);
    rocke_signature_builder_ptr(sb, "RunningMean", "f32", NULL);
    rocke_signature_builder_ptr(sb, "RunningVar", "f32", NULL);
    if(spec->save_mean_invstd)
    {
        rocke_signature_builder_ptr(sb, "Mean", spec->dtype, NULL);
        rocke_signature_builder_ptr(sb, "InvStd", spec->dtype, NULL);
    }
    rocke_signature_builder_scalar(sb, "N", "i32");
    rocke_signature_builder_scalar(sb, "C", "i32");
    rocke_signature_builder_scalar(sb, "H", "i32");
    rocke_signature_builder_scalar(sb, "W", "i32");
    rocke_signature_builder_scalar(sb, "eps", "f32");
    rocke_signature_builder_scalar(sb, "momentum", "f32");
    return rocke_signature_builder_build(sb, out_items, out_count);
}

/* ------------------------------------------------------------------ *
 * lower-to-.ll convenience
 * ------------------------------------------------------------------ */

rocke_status_t rocke_batchnorm2d_lower_to_llvm(const rocke_batchnorm2d_spec_t* spec,
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
        if(err != NULL && err_cap > 0)
        {
            bn_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_batchnorm2d_new(&b, spec);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            bn_set_reason(err, err_cap, m != NULL ? m : "build_batchnorm2d failed");
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

/* ===================================================================== *
 * Cross-CTA spatial-split reduction (two-kernel, workspace-based)
 *
 * C99 port of batchnorm2d.py's "Cross-CTA spatial-split reduction" section
 * -- see that module docstring for the full design rationale. Byte-
 * faithful reproduction of build_batchnorm2d_reduce_partial /
 * build_batchnorm2d_combine_apply's builder-call sequences.
 * ===================================================================== */

rocke_batchnorm2d_split_spec_t rocke_batchnorm2d_split_spec_default(void)
{
    rocke_batchnorm2d_split_spec_t s;
    s.n = 0;
    s.c = 0;
    s.h = 0;
    s.w = 0;
    s.split_hw = 0;
    s.block_size = 256;
    s.vec = 4;
    s.dtype = "f16";
    s.channels_per_block = 1;
    s.fast_reduce = false;
    s.wave_size = 64;
    s.name = "rocke_batchnorm2d_split";
    return s;
}

int rocke_batchnorm2d_split_hw(const rocke_batchnorm2d_split_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->h * spec->w;
}

int rocke_batchnorm2d_split_hw_chunk(const rocke_batchnorm2d_split_spec_t* spec)
{
    int hw;
    if(spec == NULL || spec->split_hw <= 0)
    {
        return 0;
    }
    hw = rocke_batchnorm2d_split_hw(spec);
    return (hw + spec->split_hw - 1) / spec->split_hw;
}

int rocke_batchnorm2d_split_workspace_floats(const rocke_batchnorm2d_split_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->c * spec->split_hw * 3;
}

static rocke_status_t bn_split_kernel_name(const rocke_batchnorm2d_split_spec_t* spec,
                                           const char* suffix,
                                           char* out,
                                           size_t out_cap)
{
    char part_dims[64];
    char part_b[32];
    char part_v[32];
    char part_split[32];
    char part_cpb[32];
    const char* parts[6];
    const char* flag_names[1];
    int flag_on[1];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    snprintf(part_dims, sizeof part_dims, "N%dC%dH%dW%d", spec->n, spec->c, spec->h, spec->w);
    snprintf(part_b, sizeof part_b, "b%d", spec->block_size);
    snprintf(part_v, sizeof part_v, "v%d", spec->vec);
    snprintf(part_split, sizeof part_split, "split%d", spec->split_hw);
    if(spec->channels_per_block != 1)
    {
        snprintf(part_cpb, sizeof part_cpb, "cpb%d", spec->channels_per_block);
    }
    else
    {
        part_cpb[0] = '\0';
    }

    parts[0] = spec->dtype;
    parts[1] = part_dims;
    parts[2] = part_b;
    parts[3] = part_v;
    parts[4] = part_split;
    parts[5] = part_cpb;

    flag_names[0] = "fr";
    flag_on[0] = spec->fast_reduce ? 1 : 0;

    /* `suffix` ("reduce"/"combine") is itself a plain part, appended after
     * the flag-eligible fields -- matching the Python builder's
     * kernel_name_join(self.name, dtype, dims, b, v, split, cpb, suffix,
     * flags={"fr": ...}) call, where positional parts precede flags in the
     * joined string regardless of argument order (flags are a separate
     * trailing pass over the dict, not interleaved with *parts). To keep
     * `suffix` in that exact trailing position next to (not before) the
     * flag suffix, it is passed as a 7th part here via a second call. */
    {
        char body[200];
        rocke_status_t st = rocke_kernel_name_join(
            spec->name, parts, 6, NULL, NULL, 0, body, sizeof body, NULL);
        if(st != ROCKE_OK)
        {
            return st;
        }
        {
            const char* parts2[2];
            parts2[0] = body;
            parts2[1] = suffix;
            return rocke_kernel_name_join(
                NULL, parts2, 2, flag_names, flag_on, 1, out, out_cap, NULL);
        }
    }
}

static void bn_split_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

bool rocke_batchnorm2d_is_valid_split_spec(const rocke_batchnorm2d_split_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap)
{
    const rocke_archtarget_t* target;
    rocke_io_spec_rule_t rule;
    rocke_arena_t arena;
    const char* why = NULL;
    int io_ok;
    int max_thr;
    long bytes_lds;

    if(reason != NULL && reason_cap > 0)
    {
        reason[0] = '\0';
    }
    if(spec == NULL)
    {
        bn_split_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        bn_split_set_reason(reason, reason_cap, "unknown arch");
        return false;
    }

    if(spec->n <= 0 || spec->c <= 0 || spec->h <= 0 || spec->w <= 0)
    {
        bn_split_set_reason(reason, reason_cap, "n, c, h, w must all be positive");
        return false;
    }
    if(spec->split_hw <= 1)
    {
        bn_split_set_reason(reason, reason_cap,
                            "split_hw must be > 1 (use the single-kernel path otherwise)");
        return false;
    }
    if(spec->channels_per_block <= 0)
    {
        bn_split_set_reason(reason, reason_cap, "channels_per_block must be positive");
        return false;
    }
    if(spec->c % spec->channels_per_block != 0)
    {
        bn_split_set_reason(reason, reason_cap, "c not divisible by channels_per_block");
        return false;
    }
    if(spec->split_hw > spec->block_size)
    {
        bn_split_set_reason(reason, reason_cap,
                            "split_hw > block_size: the combine kernel gives each thread "
                            "at most one workspace slot");
        return false;
    }
    if(spec->fast_reduce && spec->block_size % spec->wave_size != 0)
    {
        bn_split_set_reason(reason, reason_cap,
                            "fast_reduce requires block_size divisible by wave_size");
        return false;
    }

    rocke_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec);
    rule.n_per_block_set = 0; /* every HW chunk is masked, not exact-fit */

    if(rocke_arena_init(&arena, 4096) != ROCKE_OK)
    {
        bn_split_set_reason(reason, reason_cap, "arena init failed");
        return false;
    }
    io_ok = rocke_validate_io(&arena, &rule, &why);
    if(!io_ok)
    {
        bn_split_set_reason(reason, reason_cap, why != NULL ? why : "validate_io failed");
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    max_thr = rocke_archtarget_max_threads_per_block(target);
    if(spec->block_size > max_thr)
    {
        bn_split_set_reason(reason, reason_cap, "block_size > max_threads_per_block");
        return false;
    }

    bytes_lds = (long)3 * (long)spec->block_size * 4L;
    if(!rocke_archtarget_fits_lds(target, bytes_lds))
    {
        bn_split_set_reason(reason, reason_cap, "LDS budget exceeds cap");
        return false;
    }

    return true;
}

rocke_status_t rocke_batchnorm2d_reduce_partial_grid(
    int c, const rocke_batchnorm2d_split_spec_t* spec, int out[3])
{
    if(out == NULL || spec == NULL || spec->channels_per_block <= 0)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = c / spec->channels_per_block;
    out[1] = spec->split_hw;
    out[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_batchnorm2d_combine_apply_grid(
    int c, const rocke_batchnorm2d_split_spec_t* spec, int out[3])
{
    if(out == NULL || spec == NULL || spec->channels_per_block <= 0)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = c / spec->channels_per_block;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * build_batchnorm2d_reduce_partial (kernel A)
 * ------------------------------------------------------------------ */

rocke_kernel_def_t* rocke_build_batchnorm2d_reduce_partial(
    rocke_ir_builder_t* b, const rocke_batchnorm2d_split_spec_t* spec)
{
    const rocke_type_t* io_ty;
    int BS, VEC, N, C, HW, hw_chunk, chunks_per_thread, CPB;
    bool ok;

    rocke_value_t* X;
    rocke_value_t* Workspace;
    rocke_value_t* tid;
    rocke_value_t* base_channel;
    rocke_value_t* split_id;
    rocke_value_t* lds_mean = NULL;
    rocke_value_t* lds_m2 = NULL;
    rocke_value_t* lds_count = NULL;
    rocke_value_t* hw_start;
    rocke_value_t* c_hw;
    rocke_value_t* remaining;
    rocke_value_t* valid_len;
    rocke_value_t* total_count;
    rocke_param_opts_t opts;

    rocke_tensor_view_t lds_mean_v, lds_m2_v, lds_count_v;
    int ldsshape[1];
    int cb;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    ok = rocke_batchnorm2d_is_valid_split_spec(spec, "gfx950", NULL, 0);
    if(!ok)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "invalid batchnorm2d split spec");
        return NULL;
    }

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    BS = spec->block_size;
    VEC = spec->vec;
    N = spec->n;
    C = spec->c;
    HW = rocke_batchnorm2d_split_hw(spec);
    hw_chunk = rocke_batchnorm2d_split_hw_chunk(spec);
    chunks_per_thread = (hw_chunk + BS * VEC - 1) / (BS * VEC);
    CPB = spec->channels_per_block;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    X = rocke_b_param(b, "X", rocke_ptr_type(b, io_ty, "global"), &opts);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    Workspace = rocke_b_param(b, "Workspace", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    (void)rocke_b_param(b, "N", rocke_i32(), NULL);
    (void)rocke_b_param(b, "C", rocke_i32(), NULL);
    (void)rocke_b_param(b, "H", rocke_i32(), NULL);
    (void)rocke_b_param(b, "W", rocke_i32(), NULL);

    tid = rocke_b_thread_id_x(b);
    base_channel = rocke_b_block_id_x(b);
    split_id = rocke_b_block_id_y(b);

    ldsshape[0] = BS;
    rocke_make_lds_view(b, &lds_mean_v, rocke_f32(), ldsshape, 1, "lds_mean", NULL);
    rocke_make_lds_view(b, &lds_m2_v, rocke_f32(), ldsshape, 1, "lds_m2", NULL);
    rocke_make_lds_view(b, &lds_count_v, rocke_f32(), ldsshape, 1, "lds_count", NULL);
    lds_mean = lds_mean_v.base;
    lds_m2 = lds_m2_v.base;
    lds_count = lds_count_v.base;

    hw_start = rocke_b_mul(b, split_id, rocke_b_const_i32(b, hw_chunk));
    c_hw = rocke_b_const_i32(b, HW);

    remaining = rocke_b_sub(b, c_hw, hw_start);
    valid_len = rocke_b_smin(b, remaining, rocke_b_const_i32(b, hw_chunk));
    total_count = rocke_b_sitofp_f32(b, rocke_b_mul(b, rocke_b_const_i32(b, N), valid_len));

    for(cb = 0; cb < CPB; ++cb)
    {
        rocke_value_t* channel;
        rocke_value_t* sum_p;
        rocke_value_t* sumsq_p;
        rocke_value_t* count_p;
        int n;

        if(CPB == 1)
        {
            channel = base_channel;
        }
        else
        {
            rocke_value_t* scaled = rocke_b_mul(b, base_channel, rocke_b_const_i32(b, CPB));
            rocke_value_t* cb_const = rocke_b_const_i32(b, cb);
            channel = rocke_b_add(b, scaled, cb_const);
            if(cb > 0)
            {
                rocke_b_sync(b);
            }
        }

        sum_p = rocke_b_const_f32(b, 0.0);
        sumsq_p = rocke_b_const_f32(b, 0.0);
        count_p = rocke_b_const_f32(b, 0.0);

        for(n = 0; n < N; ++n)
        {
            rocke_value_t* row_n = rocke_b_add(b, rocke_b_const_i32(b, n * C), channel);
            rocke_value_t* row_base = rocke_b_mul(b, row_n, c_hw);
            int k, i;
            for(k = 0; k < chunks_per_thread; ++k)
            {
                for(i = 0; i < VEC; ++i)
                {
                    /* Match Python's left-to-right argument evaluation at
                     * every call below (C/C++ argument evaluation order is
                     * unspecified): each operand is materialised into its
                     * own statement in emission order rather than nested in
                     * the enclosing call. */
                    rocke_value_t* c_off = rocke_b_const_i32(b, k * BS * VEC + i);
                    rocke_value_t* c_vec = rocke_b_const_i32(b, VEC);
                    rocke_value_t* tid_vec = rocke_b_mul(b, tid, c_vec);
                    rocke_value_t* local = rocke_b_add(b, c_off, tid_vec);
                    rocke_value_t* idx_local = rocke_b_add(b, hw_start, local);
                    rocke_value_t* c_hw_chunk = rocke_b_const_i32(b, hw_chunk);
                    rocke_value_t* lt_local = rocke_b_cmp_lt(b, local, c_hw_chunk);
                    rocke_value_t* lt_abs = rocke_b_cmp_lt(b, idx_local, c_hw);
                    rocke_value_t* valid = rocke_b_land(b, lt_local, lt_abs);
                    rocke_value_t* idx_safe =
                        rocke_b_smin(b, idx_local, rocke_b_const_i32(b, HW - 1));
                    rocke_value_t* flat = rocke_b_add(b, row_base, idx_safe);
                    rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                    rocke_value_t* x_masked =
                        rocke_b_select(b, valid, x_raw, rocke_b_const_f32(b, 0.0));
                    rocke_value_t* c_one = rocke_b_const_f32(b, 1.0);
                    rocke_value_t* c_zero = rocke_b_const_f32(b, 0.0);
                    rocke_value_t* valid_f = rocke_b_select(b, valid, c_one, c_zero);
                    sum_p = rocke_b_fadd(b, sum_p, x_masked);
                    sumsq_p = rocke_b_fadd(b, sumsq_p, rocke_b_fmul(b, x_masked, x_masked));
                    count_p = rocke_b_fadd(b, count_p, valid_f);
                }
            }
        }

        {
            rocke_value_t* is_empty = rocke_b_fcmp(b, "oeq", count_p, rocke_b_const_f32(b, 0.0));
            rocke_value_t* safe_count =
                rocke_b_select(b, is_empty, rocke_b_const_f32(b, 1.0), count_p);
            rocke_value_t* mean_p = rocke_b_fdiv(b, sum_p, safe_count);
            rocke_value_t* m2_zero = rocke_b_const_f32(b, 0.0);
            rocke_value_t* m2_fmul = rocke_b_fmul(b, mean_p, sum_p);
            rocke_value_t* m2_fsub = rocke_b_fsub(b, sumsq_p, m2_fmul);
            rocke_value_t* m2_p = rocke_b_select(b, is_empty, m2_zero, m2_fsub);
            mean_p = rocke_b_select(b, is_empty, rocke_b_const_f32(b, 0.0), mean_p);

            rocke_value_t* mean;
            rocke_value_t* var;
            if(spec->fast_reduce)
            {
                rocke_welford_block_reduce_stable_wave_prologue(
                    b, mean_p, m2_p, count_p, lds_mean, lds_m2, lds_count, tid,
                    BS, spec->wave_size, &mean, &var);
            }
            else
            {
                rocke_welford_block_reduce_stable(
                    b, mean_p, m2_p, count_p, lds_mean, lds_m2, lds_count, tid, BS,
                    &mean, &var);
            }

            {
                rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
                rocke_value_t* slot;
                rocke_value_t* m2_out;

                rocke_b_region_enter(b, iff.then_region);
                {
                    rocke_value_t* c_split = rocke_b_const_i32(b, spec->split_hw);
                    rocke_value_t* ch_split = rocke_b_mul(b, channel, c_split);
                    rocke_value_t* slot_base = rocke_b_add(b, ch_split, split_id);
                    rocke_value_t* c_three = rocke_b_const_i32(b, 3);
                    slot = rocke_b_mul(b, slot_base, c_three);
                }
                m2_out = rocke_b_fmul(b, var, total_count);
                rocke_b_global_store(b, Workspace, slot, mean, 1);
                rocke_b_global_store(b, Workspace, rocke_b_add(b, slot, rocke_b_const_i32(b, 1)), m2_out, 1);
                rocke_b_global_store(
                    b, Workspace, rocke_b_add(b, slot, rocke_b_const_i32(b, 2)), total_count, 1);
                rocke_b_region_leave(b);
            }
        }
    }

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_batchnorm2d_reduce_partial_new(
    rocke_ir_builder_t* b, const rocke_batchnorm2d_split_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(bn_split_kernel_name(spec, "reduce", name, sizeof name) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_batchnorm2d_reduce_partial(b, spec);
    });
}

/* ------------------------------------------------------------------ *
 * build_batchnorm2d_combine_apply (kernel B)
 * ------------------------------------------------------------------ */

rocke_kernel_def_t* rocke_build_batchnorm2d_combine_apply(
    rocke_ir_builder_t* b, const rocke_batchnorm2d_split_spec_t* spec)
{
    const rocke_type_t* io_ty;
    int BS, VEC, N, C, HW, CPB, split_hw;
    int vec_bs, chunks_main, ept_row_main, main_elems_per_row, tail_per_row, tail_chunks;
    bool ok;

    rocke_value_t* X;
    rocke_value_t* Workspace;
    rocke_value_t* Gamma;
    rocke_value_t* Beta;
    rocke_value_t* Y;
    rocke_value_t* RunningMean;
    rocke_value_t* RunningVar;
    rocke_value_t* eps;
    rocke_value_t* momentum;
    rocke_value_t* tid;
    rocke_value_t* base_channel;
    rocke_value_t* lds_mean = NULL;
    rocke_value_t* lds_m2 = NULL;
    rocke_value_t* lds_count = NULL;
    rocke_param_opts_t opts;

    rocke_tensor_view_t x_view, y_view;
    rocke_tensor_view_t lds_mean_v, lds_m2_v, lds_count_v;
    int shape2[2];
    int lengths2[2];
    rocke_value_t* origin[2];
    int ldsshape[1];
    int cb;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    ok = rocke_batchnorm2d_is_valid_split_spec(spec, "gfx950", NULL, 0);
    if(!ok)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "invalid batchnorm2d split spec");
        return NULL;
    }

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    BS = spec->block_size;
    VEC = spec->vec;
    N = spec->n;
    C = spec->c;
    HW = rocke_batchnorm2d_split_hw(spec);
    CPB = spec->channels_per_block;
    split_hw = spec->split_hw;

    vec_bs = BS * VEC;
    chunks_main = HW / vec_bs;
    ept_row_main = chunks_main * VEC;
    main_elems_per_row = ept_row_main * BS;
    tail_per_row = HW - main_elems_per_row;
    tail_chunks = tail_per_row ? (tail_per_row + BS - 1) / BS : 0;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    X = rocke_b_param(b, "X", rocke_ptr_type(b, io_ty, "global"), &opts);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    Workspace = rocke_b_param(b, "Workspace", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    Gamma = rocke_b_param(b, "Gamma", rocke_ptr_type(b, io_ty, "global"), &opts);
    Beta = rocke_b_param(b, "Beta", rocke_ptr_type(b, io_ty, "global"), &opts);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    Y = rocke_b_param(b, "Y", rocke_ptr_type(b, io_ty, "global"), &opts);

    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    RunningMean = rocke_b_param(b, "RunningMean", rocke_ptr_type(b, rocke_f32(), "global"), &opts);
    RunningVar = rocke_b_param(b, "RunningVar", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    (void)rocke_b_param(b, "N", rocke_i32(), NULL);
    (void)rocke_b_param(b, "C", rocke_i32(), NULL);
    (void)rocke_b_param(b, "H", rocke_i32(), NULL);
    (void)rocke_b_param(b, "W", rocke_i32(), NULL);
    eps = rocke_b_param(b, "eps", rocke_f32(), NULL);
    momentum = rocke_b_param(b, "momentum", rocke_f32(), NULL);

    tid = rocke_b_thread_id_x(b);
    base_channel = rocke_b_block_id_x(b);

    shape2[0] = N * C;
    shape2[1] = HW;
    rocke_make_naive_tensor_view_packed(&x_view, X, shape2, 2, io_ty);
    rocke_make_naive_tensor_view_packed(&y_view, Y, shape2, 2, io_ty);

    ldsshape[0] = BS;
    rocke_make_lds_view(b, &lds_mean_v, rocke_f32(), ldsshape, 1, "lds_mean", NULL);
    rocke_make_lds_view(b, &lds_m2_v, rocke_f32(), ldsshape, 1, "lds_m2", NULL);
    rocke_make_lds_view(b, &lds_count_v, rocke_f32(), ldsshape, 1, "lds_count", NULL);
    lds_mean = lds_mean_v.base;
    lds_m2 = lds_m2_v.base;
    lds_count = lds_count_v.base;

    lengths2[0] = 1;
    lengths2[1] = HW;

    for(cb = 0; cb < CPB; ++cb)
    {
        rocke_value_t* channel;
        rocke_value_t* has_slot;
        rocke_value_t* slot_tid;
        rocke_value_t* slot;
        rocke_value_t* mean_raw;
        rocke_value_t* m2_raw;
        rocke_value_t* count_raw;
        rocke_value_t* mean_p;
        rocke_value_t* m2_p;
        rocke_value_t* count_p;
        rocke_value_t* mean = NULL;
        rocke_value_t* var = NULL;
        rocke_value_t* inv_std;
        rocke_value_t* gamma_v;
        rocke_value_t* beta_v;
        rocke_value_t* scale;
        int n;

        if(CPB == 1)
        {
            channel = base_channel;
        }
        else
        {
            rocke_value_t* scaled = rocke_b_mul(b, base_channel, rocke_b_const_i32(b, CPB));
            rocke_value_t* cb_const = rocke_b_const_i32(b, cb);
            channel = rocke_b_add(b, scaled, cb_const);
            if(cb > 0)
            {
                rocke_b_sync(b);
            }
        }

        has_slot = rocke_b_cmp_lt(b, tid, rocke_b_const_i32(b, split_hw));
        slot_tid = rocke_b_select(b, has_slot, tid, rocke_b_const_i32(b, 0));
        {
            rocke_value_t* c_split = rocke_b_const_i32(b, split_hw);
            rocke_value_t* ch_split = rocke_b_mul(b, channel, c_split);
            rocke_value_t* slot_base = rocke_b_add(b, ch_split, slot_tid);
            rocke_value_t* c_three = rocke_b_const_i32(b, 3);
            slot = rocke_b_mul(b, slot_base, c_three);
        }
        mean_raw = rocke_b_global_load_f32(b, Workspace, slot, 4);
        m2_raw = rocke_b_global_load_f32(b, Workspace, rocke_b_add(b, slot, rocke_b_const_i32(b, 1)), 4);
        count_raw = rocke_b_global_load_f32(b, Workspace, rocke_b_add(b, slot, rocke_b_const_i32(b, 2)), 4);
        mean_p = rocke_b_select(b, has_slot, mean_raw, rocke_b_const_f32(b, 0.0));
        m2_p = rocke_b_select(b, has_slot, m2_raw, rocke_b_const_f32(b, 0.0));
        count_p = rocke_b_select(b, has_slot, count_raw, rocke_b_const_f32(b, 0.0));

        if(spec->fast_reduce)
        {
            rocke_welford_block_reduce_stable_wave_prologue(
                b, mean_p, m2_p, count_p, lds_mean, lds_m2, lds_count, tid,
                BS, spec->wave_size, &mean, &var);
        }
        else
        {
            rocke_welford_block_reduce_stable(
                b, mean_p, m2_p, count_p, lds_mean, lds_m2, lds_count, tid, BS, &mean, &var);
        }

        {
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
            rocke_value_t* old_mean;
            rocke_value_t* old_var;
            rocke_value_t* one_minus_mom;
            rocke_value_t* new_running_mean;
            rocke_value_t* new_running_var;

            rocke_b_region_enter(b, iff.then_region);
            old_mean = rocke_b_global_load_f32(b, RunningMean, channel, 4);
            old_var = rocke_b_global_load_f32(b, RunningVar, channel, 4);
            one_minus_mom = rocke_b_fsub(b, rocke_b_const_f32(b, 1.0), momentum);
            {
                rocke_value_t* term_a = rocke_b_fmul(b, one_minus_mom, old_mean);
                rocke_value_t* term_b = rocke_b_fmul(b, momentum, mean);
                new_running_mean = rocke_b_fadd(b, term_a, term_b);
            }
            {
                rocke_value_t* term_a = rocke_b_fmul(b, one_minus_mom, old_var);
                rocke_value_t* term_b = rocke_b_fmul(b, momentum, var);
                new_running_var = rocke_b_fadd(b, term_a, term_b);
            }
            rocke_b_global_store(b, RunningMean, channel, new_running_mean, 1);
            rocke_b_global_store(b, RunningVar, channel, new_running_var, 1);
            rocke_b_region_leave(b);
        }

        inv_std = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));
        gamma_v = rocke_b_load_scalar_as_f32(b, Gamma, channel, spec->dtype);
        beta_v = rocke_b_load_scalar_as_f32(b, Beta, channel, spec->dtype);
        scale = rocke_b_fmul(b, inv_std, gamma_v);

        for(n = 0; n < N; ++n)
        {
            rocke_value_t* row_n = rocke_b_add(b, rocke_b_const_i32(b, n * C), channel);
            rocke_tile_window_t y_tile_n;
            rocke_tile_window_t x_tile_n;
            bn_pass2_ctx_t p2;
            int tc;

            origin[0] = row_n;
            origin[1] = rocke_b_const_i32(b, 0);
            rocke_make_tile_window(&y_tile_n, &y_view, lengths2, origin, 2);
            origin[0] = row_n;
            origin[1] = rocke_b_const_i32(b, 0);
            rocke_make_tile_window(&x_tile_n, &x_view, lengths2, origin, 2);

            p2.two_pass = true;
            p2.training = true;
            p2.vec = VEC;
            p2.x_tile_n = &x_tile_n;
            p2.mean = mean;
            p2.scale = scale;
            p2.beta_v = beta_v;

            rocke_pass2_row_chunks(b, &y_tile_n, tid, BS, VEC, ept_row_main, NULL,
                                   bn_pass2_body, &p2, NULL, 0);

            for(tc = 0; tc < tail_chunks; ++tc)
            {
                rocke_value_t* idx_col =
                    rocke_b_add(b, rocke_b_const_i32(b, main_elems_per_row + tc * BS), tid);
                rocke_value_t* valid_tail = rocke_b_cmp_lt(b, idx_col, rocke_b_const_i32(b, HW));
                rocke_if_t tail_iff = rocke_b_scf_if(b, valid_tail);
                rocke_value_t* flat;
                rocke_value_t* x_raw;
                rocke_value_t* y_val;

                rocke_b_region_enter(b, tail_iff.then_region);
                flat = rocke_b_add(b, rocke_b_mul(b, row_n, rocke_b_const_i32(b, HW)), idx_col);
                x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
                y_val = rocke_b_fadd(b, rocke_b_fmul(b, rocke_b_fsub(b, x_raw, mean), scale), beta_v);
                rocke_b_store_scalar_from_f32(b, Y, flat, y_val, spec->dtype);
                rocke_b_region_leave(b);
            }
        }
    }

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_batchnorm2d_combine_apply_new(
    rocke_ir_builder_t* b, const rocke_batchnorm2d_split_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(bn_split_kernel_name(spec, "combine", name, sizeof name) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_batchnorm2d_combine_apply(b, spec);
    });
}
