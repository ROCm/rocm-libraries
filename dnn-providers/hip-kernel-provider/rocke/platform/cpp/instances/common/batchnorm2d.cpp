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
    const char* flag_names[3];
    int flag_on[3];

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

    return rocke_kernel_name_join(spec->name, parts, 6, flag_names, flag_on, 3, out, out_cap, NULL);
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
    int hw;
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
    if(spec->c % spec->channels_per_block != 0)
    {
        bn_set_reason(reason, reason_cap, "c not divisible by channels_per_block");
        return false;
    }

    if(spec->n_groups <= 0)
    {
        bn_set_reason(reason, reason_cap, "n_groups must be positive");
        return false;
    }
    if(spec->n_groups != 1 && spec->training)
    {
        bn_set_reason(reason, reason_cap, "n_groups > 1 requires training=False");
        return false;
    }
    if(spec->block_size % spec->n_groups != 0)
    {
        bn_set_reason(reason, reason_cap, "block_size not divisible by n_groups");
        return false;
    }
    if(spec->n % spec->n_groups != 0)
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

    hw = rocke_batchnorm2d_hw(spec);

    /* validate_io(IOSpecRule(dtype, block_size=hw_threads, vec, n_per_block=hw)),
     * where hw_threads = block_size // n_groups (the per-sub-group thread
     * count the n_groups>1 path actually sweeps HW with). */
    rocke_io_spec_rule_init(&rule, spec->dtype, spec->block_size / spec->n_groups, spec->vec);
    rule.n_per_block_set = 1;
    rule.n_per_block = hw;
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

    elems = rocke_batchnorm2d_elems_per_thread(spec);
    two_pass = rocke_row_norm_needs_two_pass(elems, ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD);
    if(!two_pass && elems > ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD)
    {
        bn_set_reason(reason, reason_cap, "elems_per_thread exceeds register-tile cap");
        return false;
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
    ept_row = HW / BS;
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
                }
            }

            if(spec->training)
            {
                double count_p = (double)rocke_batchnorm2d_elems_per_thread(spec);
                rocke_value_t* inv_count_p = rocke_b_const_f32(b, 1.0 / count_p);
                rocke_value_t* mean_p = rocke_b_fmul(b, p1.sum_p, inv_count_p);
                rocke_value_t* m2_p =
                    rocke_b_fsub(b, p1.sumsq_p, rocke_b_fmul(b, mean_p, p1.sum_p));

                if(spec->fast_reduce)
                {
                    rocke_welford_block_reduce_stable_wave_prologue(b,
                                                                    mean_p,
                                                                    m2_p,
                                                                    rocke_b_const_f32(b, count_p),
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
                                                      rocke_b_const_f32(b, count_p),
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
            }
            }
            else
            {
                /* n_groups > 1: inference-only, N and H*W split across the
                 * CTA's threads simultaneously (mirrors MIOpen's spatial
                 * batchnorm thread mapping). is_valid_spec guarantees
                 * training=false here, so there is no Welford/LDS state to
                 * thread across the sub-group boundary -- just the scalar
                 * running-stat read and the elementwise affine, applied
                 * with a different thread decomposition than the NG==1
                 * inference branch above. */
                int hw_threads;
                int ept_row_ng;
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

                hw_threads = BS / NG;
                ept_row_ng = HW / hw_threads;
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
