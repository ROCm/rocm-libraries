// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_layernorm2d.c -- C99 port of
 * rocke/instances/common/layernorm2d.py.
 *
 * Byte-faithful reproduction of build_layernorm2d()'s builder-call sequence.
 * See instance_layernorm2d.h for the public surface.
 */

#include "rocke/instance_layernorm2d.h"

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

/* The tensor_view f32 / lds factory peers --
 * make_naive_tensor_view_packed, make_lds_view, TensorView.load_vec_as_f32 --
 * are declared in helper_rocke.helpers.tensor_view.h and defined in that
 * module's translation unit. The TileWindow.load_vec_as_f32 /
 * store_vec_from_f32 peers come from helper_rocke.helpers.sweep.h. */

/* ------------------------------------------------------------------ *
 * spec defaults / properties
 * ------------------------------------------------------------------ */

rocke_layernorm2d_spec_t rocke_layernorm2d_spec_default(void)
{
    rocke_layernorm2d_spec_t s;
    s.n_per_block = 0; /* required: caller must set */
    s.block_size = 256;
    s.vec = 4;
    s.dtype = "f16";
    s.save_mean_invstd = false;
    s.wave_size = 64;
    s.dynamic = false;
    s.fast_reduce = false;
    s.cheap_reduce = false;
    s.name = "rocke_layernorm2d_fwd";
    return s;
}

int rocke_layernorm2d_elems_per_thread(const rocke_layernorm2d_spec_t* spec)
{
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return spec->n_per_block / spec->block_size;
}

/* LayerNorm2DSpec.kernel_name():
 *   kernel_name_join(self.name, self.dtype, f"N{n_per_block}", f"b{block_size}",
 *                    f"v{vec}", flags={"smv": save_mean_invstd, "dyn": dynamic,
 *                    "fr": fast_reduce, "cr": cheap_reduce}) */
rocke_status_t
    rocke_layernorm2d_kernel_name(const rocke_layernorm2d_spec_t* spec, char* out, size_t out_cap)
{
    char part_n[32];
    char part_b[32];
    char part_v[32];
    const char* parts[4];
    const char* flag_names[4];
    int flag_on[4];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    snprintf(part_n, sizeof part_n, "N%d", spec->n_per_block);
    snprintf(part_b, sizeof part_b, "b%d", spec->block_size);
    snprintf(part_v, sizeof part_v, "v%d", spec->vec);

    parts[0] = spec->dtype;
    parts[1] = part_n;
    parts[2] = part_b;
    parts[3] = part_v;

    flag_names[0] = "smv";
    flag_on[0] = spec->save_mean_invstd ? 1 : 0;
    flag_names[1] = "dyn";
    flag_on[1] = spec->dynamic ? 1 : 0;
    flag_names[2] = "fr";
    flag_on[2] = spec->fast_reduce ? 1 : 0;
    flag_names[3] = "cr";
    flag_on[3] = spec->cheap_reduce ? 1 : 0;

    return rocke_kernel_name_join(spec->name, parts, 4, flag_names, flag_on, 4, out, out_cap, NULL);
}

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ */

static void ln_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

bool rocke_layernorm2d_is_valid_spec(const rocke_layernorm2d_spec_t* spec,
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
        ln_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* target = ArchTarget.from_gfx(arch)  (KeyError -> reject). */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        ln_set_reason(reason, reason_cap, "unknown arch");
        return false;
    }

    elems = rocke_layernorm2d_elems_per_thread(spec);

    /* cap = None if row_norm_needs_two_pass(elems) else
     *       REGISTER_TILE_MAX_ELEMS_PER_THREAD */
    two_pass = rocke_row_norm_needs_two_pass(elems, ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD);

    /* validate_io(IOSpecRule(dtype, block_size, vec, n_per_block, cap)). */
    rocke_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec);
    if(spec->dynamic)
    {
        /* Dynamic always streams X twice (no per-row register cache), and
         * n_per_block is a capacity, not the real row width, so its exact
         * divisibility check does not apply. */
        rule.n_per_block_set = 0;
        rule.max_elems_per_thread_set = 0;
    }
    else
    {
        rule.n_per_block_set = 1;
        rule.n_per_block = spec->n_per_block;
        if(two_pass)
        {
            rule.max_elems_per_thread_set = 0; /* None */
        }
        else
        {
            rule.max_elems_per_thread_set = 1;
            rule.max_elems_per_thread = ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD;
        }
    }

    if(rocke_arena_init(&arena, 4096) != ROCKE_OK)
    {
        ln_set_reason(reason, reason_cap, "arena init failed");
        return false;
    }
    io_ok = rocke_validate_io(&arena, &rule, &why);
    if(!io_ok)
    {
        ln_set_reason(reason, reason_cap, why != NULL ? why : "validate_io failed");
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    /* if block_size > target.max_threads_per_block: reject */
    max_thr = rocke_archtarget_max_threads_per_block(target);
    if(spec->block_size > max_thr)
    {
        ln_set_reason(reason, reason_cap, "block_size > max_threads_per_block");
        return false;
    }

    if(spec->fast_reduce && spec->block_size % spec->wave_size != 0)
    {
        ln_set_reason(reason, reason_cap, "fast_reduce requires block_size divisible by wave_size");
        return false;
    }

    /* Three f32 Welford reduction buffers: 3 * block_size * 4 bytes. */
    bytes_lds = (long)3 * (long)spec->block_size * 4L;
    if(!rocke_archtarget_fits_lds(target, bytes_lds))
    {
        ln_set_reason(reason, reason_cap, "LDS budget exceeds cap");
        return false;
    }

    return true;
}

/* ------------------------------------------------------------------ *
 * tree_reduce combiner cookie: forward b.fadd as a rocke_combine_fn.
 * ------------------------------------------------------------------ */

static rocke_value_t*
    ln_fadd_combine(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, void* user)
{
    (void)user;
    return rocke_b_fadd(b, a, c);
}

/* ------------------------------------------------------------------ *
 * pass-1 / pass-2 body contexts (Python closures over nonlocal state).
 * ------------------------------------------------------------------ */

typedef struct ln_pass1_ctx
{
    rocke_value_t* sum_p;
    rocke_value_t* sumsq_p;
} ln_pass1_ctx_t;

/* pass1_body(_n_off, x_scalars):
 *     sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
 *     sum_p   = b.fadd(sum_p,   tree_reduce(b, b.fadd, list(x_scalars)))
 *     sumsq_p = b.fadd(sumsq_p, tree_reduce(b, b.fadd, sq_scalars)) */
static void ln_pass1_body(rocke_ir_builder_t* b,
                          rocke_value_t* n_off,
                          rocke_value_t* const* x_scalars,
                          int vec,
                          void* user)
{
    ln_pass1_ctx_t* ctx = (ln_pass1_ctx_t*)user;
    /* sq_scalars (size vec); vec is one of {2,4,8}, small fixed bound. */
    rocke_value_t* sq[8];
    rocke_value_t* xs[8];
    rocke_value_t* part_sum;
    rocke_value_t* part_sumsq;
    int i;

    (void)n_off;
    if(vec > 8)
    {
        return; /* validate_io bounds vec to {2,4,8}; defensive */
    }

    for(i = 0; i < vec; ++i)
    {
        xs[i] = x_scalars[i];
        sq[i] = rocke_b_fmul(b, x_scalars[i], x_scalars[i]);
    }

    /* Python pass1_body fully evaluates the sum_p statement (x tree-reduce
     * then accumulate fadd) before the sumsq_p statement (sq tree-reduce then
     * accumulate fadd). The two accumulate fadds therefore interleave between
     * the two reduces; emit in that exact order. */
    part_sum = rocke_tree_reduce(b, ln_fadd_combine, NULL, xs, vec);
    ctx->sum_p = rocke_b_fadd(b, ctx->sum_p, part_sum);

    part_sumsq = rocke_tree_reduce(b, ln_fadd_combine, NULL, sq, vec);
    ctx->sumsq_p = rocke_b_fadd(b, ctx->sumsq_p, part_sumsq);
}

typedef struct ln_pass2_ctx
{
    bool two_pass;
    int vec;
    const rocke_tile_window_t* x_tile;
    const rocke_tensor_view_t* g_view;
    const rocke_tensor_view_t* b_view;
    rocke_value_t* mean;
    rocke_value_t* inv_std;
} ln_pass2_ctx_t;

/* pass2_body(n_off, _k, x_scalars):
 *     if two_pass:
 *         x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
 *     gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
 *     bv = b_view.load_vec_as_f32(b, [n_off], n=VEC)
 *     return [ b.fadd(
 *                  b.fmul(b.fsub(x_scalars[i], mean), b.fmul(inv_std, gv[i])),
 *                  bv[i])
 *              for i in range(VEC) ] */
static void ln_pass2_body(rocke_ir_builder_t* b,
                          rocke_value_t* n_off,
                          int k,
                          rocke_value_t* const* x_scalars,
                          int num_x,
                          rocke_value_t** out,
                          int vec,
                          void* user)
{
    ln_pass2_ctx_t* ctx = (ln_pass2_ctx_t*)user;
    rocke_value_t* xs[8];
    rocke_value_t* gv[8];
    rocke_value_t* bv[8];
    rocke_value_t* idx1[1];
    int i;

    (void)k;
    if(vec > 8)
    {
        return;
    }

    if(ctx->two_pass)
    {
        /* x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC) */
        rocke_value_t* li[2];
        li[0] = rocke_b_const_i32(b, 0);
        li[1] = n_off;
        rocke_tile_window_load_vec_as_f32(b, ctx->x_tile, li, 2, vec, xs);
    }
    else
    {
        for(i = 0; i < vec && i < num_x; ++i)
        {
            xs[i] = x_scalars[i];
        }
    }

    /* gv = g_view.load_vec_as_f32(b, [n_off], n=VEC) */
    idx1[0] = n_off;
    rocke_tensor_view_load_vec_as_f32(b, ctx->g_view, idx1, 1, vec, gv);
    /* bv = b_view.load_vec_as_f32(b, [n_off], n=VEC) */
    rocke_tensor_view_load_vec_as_f32(b, ctx->b_view, idx1, 1, vec, bv);

    for(i = 0; i < vec; ++i)
    {
        rocke_value_t* dx = rocke_b_fsub(b, xs[i], ctx->mean);
        rocke_value_t* sg = rocke_b_fmul(b, ctx->inv_std, gv[i]);
        out[i] = rocke_b_fadd(b, rocke_b_fmul(b, dx, sg), bv[i]);
    }
}

/* ------------------------------------------------------------------ *
 * build_layernorm2d_dynamic -- AOT-generic variant (runtime row width)
 *
 * C mirror of Python's _build_layernorm2d_dynamic: see that function's
 * docstring. Main sweep is a genuine scf.for (rocke_sweep_row_chunks_dynamic
 * / rocke_pass2_row_chunks_dynamic); the <block_size*vec remainder is a
 * vec-wide, C-unrolled masked scalar tail with a runtime bound (cols)
 * instead of a compile-time one.
 * ------------------------------------------------------------------ */

typedef struct ln_dyn_pass1_ctx
{
    rocke_ir_builder_t* b;
} ln_dyn_pass1_ctx_t;

static void ln_dyn_pass1_body(rocke_ir_builder_t* b,
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
    part_sum = rocke_tree_reduce(b, ln_fadd_combine, NULL, (rocke_value_t* const*)x_scalars, vec);
    new_iter_vals[0] = rocke_b_fadd(b, iter_vars[0], part_sum);
    part_sumsq = rocke_tree_reduce(b, ln_fadd_combine, NULL, sq, vec);
    new_iter_vals[1] = rocke_b_fadd(b, iter_vars[1], part_sumsq);
}

typedef struct ln_dyn_pass2_ctx
{
    const rocke_tile_window_t* x_tile;
    const rocke_tensor_view_t* g_view;
    const rocke_tensor_view_t* b_view;
    rocke_value_t* mean;
    rocke_value_t* inv_std;
} ln_dyn_pass2_ctx_t;

static void ln_dyn_pass2_body(rocke_ir_builder_t* b,
                              rocke_value_t* n_off,
                              rocke_value_t** out,
                              int vec,
                              void* user)
{
    ln_dyn_pass2_ctx_t* ctx = (ln_dyn_pass2_ctx_t*)user;
    rocke_value_t* xs[8];
    rocke_value_t* gv[8];
    rocke_value_t* bv[8];
    rocke_value_t* li[2];
    rocke_value_t* idx1[1];
    int i;

    li[0] = rocke_b_const_i32(b, 0);
    li[1] = n_off;
    rocke_tile_window_load_vec_as_f32(b, ctx->x_tile, li, 2, vec, xs);
    idx1[0] = n_off;
    rocke_tensor_view_load_vec_as_f32(b, ctx->g_view, idx1, 1, vec, gv);
    rocke_tensor_view_load_vec_as_f32(b, ctx->b_view, idx1, 1, vec, bv);

    for(i = 0; i < vec; ++i)
    {
        rocke_value_t* dx = rocke_b_fsub(b, xs[i], ctx->mean);
        rocke_value_t* sg = rocke_b_fmul(b, ctx->inv_std, gv[i]);
        out[i] = rocke_b_fadd(b, rocke_b_fmul(b, dx, sg), bv[i]);
    }
}

static rocke_kernel_def_t* rocke_build_layernorm2d_dynamic(rocke_ir_builder_t* b,
                                                            const rocke_layernorm2d_spec_t* spec)
{
    const rocke_type_t* io_ty;
    int BS, VEC, CAP;
    rocke_value_t *X, *Gamma, *Beta, *Y, *Mean = NULL, *InvStd = NULL, *cols, *eps;
    rocke_value_t *tid, *row;
    rocke_tensor_view_t x_view, y_view, g_view, b_view;
    rocke_tile_window_t x_tile, y_tile;
    rocke_tensor_view_t lds_mean_v, lds_m2_v, lds_count_v;
    rocke_value_t *lds_mean, *lds_m2, *lds_count;
    rocke_stride_t row_stride[2];
    int shape2[2];
    int shape1[1];
    int ldsshape[1];
    rocke_value_t* origin[2];
    int lengths2[2];
    rocke_param_opts_t opts;
    rocke_iter_arg_t iter_args[2];
    rocke_sweep_dynamic_result_t sweep_res;
    rocke_value_t *sum_p, *sumsq_p, *main_elems, *main_per_thread, *tail_count_p;
    rocke_value_t *count_p, *mean_p, *m2_p, *mean, *var, *inv_std;
    int tc;

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
        return NULL;
    BS = spec->block_size;
    VEC = spec->vec;
    CAP = spec->n_per_block;

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

    (void)rocke_b_param(b, "M", rocke_i32(), NULL);
    cols = rocke_b_param(b, "N", rocke_i32(), NULL);
    eps = rocke_b_param(b, "eps", rocke_f32(), NULL);

    tid = rocke_b_thread_id_x(b);
    row = rocke_b_block_id_x(b);

    shape2[0] = 1;
    shape2[1] = CAP;
    row_stride[0] = rocke_stride_value(cols);
    row_stride[1] = rocke_stride_imm(1);
    rocke_make_global_view(&x_view, X, shape2, 2, io_ty, row_stride);
    rocke_make_global_view(&y_view, Y, shape2, 2, io_ty, row_stride);
    shape1[0] = CAP;
    rocke_make_global_view(&g_view, Gamma, shape1, 1, io_ty, NULL);
    rocke_make_global_view(&b_view, Beta, shape1, 1, io_ty, NULL);

    lengths2[0] = 1;
    lengths2[1] = 1;
    origin[0] = row;
    origin[1] = rocke_b_const_i32(b, 0);
    rocke_make_tile_window(&x_tile, &x_view, lengths2, origin, 2);
    origin[0] = row;
    origin[1] = rocke_b_const_i32(b, 0);
    rocke_make_tile_window(&y_tile, &y_view, lengths2, origin, 2);

    ldsshape[0] = BS;
    rocke_make_lds_view(b, &lds_mean_v, rocke_f32(), ldsshape, 1, "lds_mean", NULL);
    rocke_make_lds_view(b, &lds_m2_v, rocke_f32(), ldsshape, 1, "lds_m2", NULL);
    rocke_make_lds_view(b, &lds_count_v, rocke_f32(), ldsshape, 1, "lds_count", NULL);
    lds_mean = lds_mean_v.base;
    lds_m2 = lds_m2_v.base;
    lds_count = lds_count_v.base;

    iter_args[0].name = "sum_p";
    iter_args[0].init = rocke_b_const_f32(b, 0.0);
    iter_args[1].name = "sumsq_p";
    iter_args[1].init = rocke_b_const_f32(b, 0.0);

    sweep_res = rocke_sweep_row_chunks_dynamic(
        b, &x_tile, tid, BS, VEC, cols, iter_args, 2, NULL, ln_dyn_pass1_body, NULL, "k0");
    sum_p = sweep_res.final_iter_vals[0];
    sumsq_p = sweep_res.final_iter_vals[1];
    main_elems = sweep_res.main_elems_per_row;

    tail_count_p = rocke_b_const_f32(b, 0.0);
    for(tc = 0; tc < VEC; ++tc)
    {
        rocke_value_t* idx_col = rocke_b_add(
            b, rocke_b_add(b, main_elems, rocke_b_const_i32(b, tc * BS)), tid);
        rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, cols);
        rocke_value_t* idx_safe =
            rocke_b_smin(b, idx_col, rocke_b_sub(b, cols, rocke_b_const_i32(b, 1)));
        rocke_value_t* flat = rocke_b_add(b, rocke_b_mul(b, row, cols), idx_safe);
        rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
        rocke_value_t* zero_f = rocke_b_const_f32(b, 0.0);
        rocke_value_t* x_masked = rocke_b_select(b, valid, x_raw, zero_f);
        rocke_value_t* one_f2 = rocke_b_const_f32(b, 1.0);
        rocke_value_t* zero_f2 = rocke_b_const_f32(b, 0.0);
        rocke_value_t* valid_f = rocke_b_select(b, valid, one_f2, zero_f2);
        sum_p = rocke_b_fadd(b, sum_p, x_masked);
        sumsq_p = rocke_b_fadd(b, sumsq_p, rocke_b_fmul(b, x_masked, x_masked));
        tail_count_p = rocke_b_fadd(b, tail_count_p, valid_f);
    }

    mean = NULL;
    var = NULL;
    if(spec->cheap_reduce)
    {
        /* MIOpen-style cheap reduction (see LayerNorm2DSpec's docstring):
         * masked tail elements already contribute exactly 0 to
         * sum_p/sumsq_p, so a plain commutative sum across all BS threads
         * totals exactly cols real elements -- dividing by the runtime
         * cols kernarg directly needs no per-thread count reduction (no
         * lds_count, no is_empty guard) at all, unlike the Welford paths
         * below. */
        rocke_value_t* total_sum = NULL;
        rocke_value_t* total_sumsq = NULL;
        if(spec->fast_reduce)
        {
            rocke_block_lds_reduce_pair_with_wave_prologue(b, sum_p, sumsq_p, lds_mean, lds_m2,
                                                           tid, BS, ROCKE_REDUCE_SUM,
                                                           ROCKE_REDUCE_SUM, spec->wave_size,
                                                           &total_sum, &total_sumsq);
        }
        else
        {
            rocke_block_lds_reduce_pair(b, sum_p, sumsq_p, lds_mean, lds_m2, tid, BS,
                                         ROCKE_REDUCE_SUM, ROCKE_REDUCE_SUM,
                                         &total_sum, &total_sumsq);
        }
        rocke_value_t* one_f32 = rocke_b_const_f32(b, 1.0);
        rocke_value_t* cols_f32 = rocke_b_sitofp_f32(b, cols);
        rocke_value_t* inv_cols = rocke_b_fdiv(b, one_f32, cols_f32);
        mean = rocke_b_fmul(b, total_sum, inv_cols);
        rocke_value_t* sq_mean = rocke_b_fmul(b, total_sumsq, inv_cols);
        var = rocke_b_fsub(b, sq_mean, rocke_b_fmul(b, mean, mean));
    }
    else
    {
        main_per_thread = rocke_b_div(b, main_elems, rocke_b_const_i32(b, BS));
        count_p = rocke_b_fadd(b, rocke_b_sitofp_f32(b, main_per_thread), tail_count_p);
        /* A thread can own zero elements when cols < BS (row shorter than the
         * block); guard the reciprocal so it reports a finite empty partial
         * (mean=0, M2=0) instead of 1/0 -> inf -> 0*inf -> NaN, which would
         * otherwise poison the whole block's welford merge. */
        rocke_value_t* is_empty_p = rocke_b_fcmp(b, "oeq", count_p, rocke_b_const_f32(b, 0.0));
        rocke_value_t* zero_recip = rocke_b_const_f32(b, 0.0);
        rocke_value_t* recip_p = rocke_b_fdiv(b, rocke_b_const_f32(b, 1.0), count_p);
        rocke_value_t* inv_count_p = rocke_b_select(b, is_empty_p, zero_recip, recip_p);
        mean_p = rocke_b_fmul(b, sum_p, inv_count_p);
        m2_p = rocke_b_fsub(b, sumsq_p, rocke_b_fmul(b, mean_p, sum_p));

        if(spec->fast_reduce)
        {
            rocke_welford_block_reduce_stable_wave_prologue(
                b, mean_p, m2_p, count_p, lds_mean, lds_m2, lds_count, tid, BS, spec->wave_size,
                &mean, &var);
        }
        else
        {
            rocke_welford_block_reduce_stable(
                b, mean_p, m2_p, count_p, lds_mean, lds_m2, lds_count, tid, BS, &mean, &var);
        }
    }
    inv_std = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));

    if(spec->save_mean_invstd)
    {
        rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
        rocke_b_region_enter(b, iff.then_region);
        rocke_b_store_scalar_from_f32(b, Mean, row, mean, spec->dtype);
        rocke_b_store_scalar_from_f32(b, InvStd, row, inv_std, spec->dtype);
        rocke_b_region_leave(b);
    }

    {
        ln_dyn_pass2_ctx_t p2;
        rocke_value_t* trip_count;
        p2.x_tile = &x_tile;
        p2.g_view = &g_view;
        p2.b_view = &b_view;
        p2.mean = mean;
        p2.inv_std = inv_std;

        trip_count = rocke_b_div(b, cols, rocke_b_const_i32(b, BS * VEC));
        rocke_pass2_row_chunks_dynamic(
            b, &y_tile, tid, BS, VEC, main_elems, trip_count, NULL, ln_dyn_pass2_body, &p2, "k0");
    }

    for(tc = 0; tc < VEC; ++tc)
    {
        rocke_value_t* idx_col = rocke_b_add(
            b, rocke_b_add(b, main_elems, rocke_b_const_i32(b, tc * BS)), tid);
        rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, cols);
        rocke_if_t iff = rocke_b_scf_if(b, valid);
        rocke_b_region_enter(b, iff.then_region);
        {
            rocke_value_t* flat = rocke_b_add(b, rocke_b_mul(b, row, cols), idx_col);
            rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
            rocke_value_t* gv1;
            rocke_value_t* bv1;
            rocke_value_t* idx1[1];
            idx1[0] = idx_col;
            rocke_tensor_view_load_vec_as_f32(b, &g_view, idx1, 1, 1, &gv1);
            rocke_tensor_view_load_vec_as_f32(b, &b_view, idx1, 1, 1, &bv1);
            {
                rocke_value_t* dx = rocke_b_fsub(b, x_raw, mean);
                rocke_value_t* sg = rocke_b_fmul(b, inv_std, gv1);
                rocke_value_t* y_val = rocke_b_fadd(b, rocke_b_fmul(b, dx, sg), bv1);
                rocke_b_store_scalar_from_f32(b, Y, flat, y_val, spec->dtype);
            }
        }
        rocke_b_region_leave(b);
    }

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

/* ------------------------------------------------------------------ *
 * build_layernorm2d
 * ------------------------------------------------------------------ */

rocke_kernel_def_t* rocke_build_layernorm2d(rocke_ir_builder_t* b,
                                            const rocke_layernorm2d_spec_t* spec)
{
    const rocke_type_t* io_ty;
    int BS;
    int VEC;
    int N;
    int elems;
    bool two_pass;
    bool ok;

    rocke_value_t* X;
    rocke_value_t* Gamma;
    rocke_value_t* Beta;
    rocke_value_t* Y;
    rocke_value_t* Mean = NULL;
    rocke_value_t* InvStd = NULL;
    rocke_value_t* eps;

    rocke_value_t* tid;
    rocke_value_t* row;

    rocke_tensor_view_t x_view;
    rocke_tensor_view_t y_view;
    rocke_tensor_view_t g_view;
    rocke_tensor_view_t b_view;
    rocke_tile_window_t x_tile;
    rocke_tile_window_t y_tile;

    rocke_tensor_view_t lds_mean_v;
    rocke_tensor_view_t lds_m2_v;
    rocke_tensor_view_t lds_count_v;
    rocke_value_t* lds_mean;
    rocke_value_t* lds_m2;
    rocke_value_t* lds_count;

    ln_pass1_ctx_t p1;
    ln_pass2_ctx_t p2;
    rocke_row_chunk_sweep_result_t sweep_res;

    double count_p;
    rocke_value_t* inv_count_p;
    rocke_value_t* mean_p;
    rocke_value_t* m2_p;
    rocke_value_t* mean;
    rocke_value_t* var;
    rocke_value_t* inv_std;

    int shape2[2];
    int shape1[1];
    int ldsshape[1];
    rocke_value_t* origin[2];
    int lengths2[2];

    rocke_param_opts_t opts;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }

    /* ok, why = is_valid_spec(spec); if not ok: raise ValueError(...) */
    ok = rocke_layernorm2d_is_valid_spec(spec, "gfx950", NULL, 0);
    if(!ok)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "invalid layernorm2d spec");
        return NULL;
    }

    if(spec->dynamic)
    {
        return rocke_build_layernorm2d_dynamic(b, spec);
    }

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    BS = spec->block_size;
    VEC = spec->vec;
    N = spec->n_per_block;
    elems = rocke_layernorm2d_elems_per_thread(spec);

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* --- params (ABI order matches CK Tile) --- */
    /* X = b.param("X", PtrType(io_ty,"global"), noalias, readonly, align=16) */
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

    /* Y = b.param("Y", PtrType(io_ty,"global"), noalias, writeonly, align=16) */
    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    Y = rocke_b_param(b, "Y", rocke_ptr_type(b, io_ty, "global"), &opts);

    if(spec->save_mean_invstd)
    {
        /* Mean / InvStd: noalias, writeonly (no align kwarg). */
        memset(&opts, 0, sizeof opts);
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        Mean = rocke_b_param(b, "Mean", rocke_ptr_type(b, io_ty, "global"), &opts);
        InvStd = rocke_b_param(b, "InvStd", rocke_ptr_type(b, io_ty, "global"), &opts);
    }

    /* M = b.param("M", I32); _ = b.param("N", I32); eps = b.param("eps", F32) */
    (void)rocke_b_param(b, "M", rocke_i32(), NULL);
    (void)rocke_b_param(b, "N", rocke_i32(), NULL);
    eps = rocke_b_param(b, "eps", rocke_f32(), NULL);

    /* tid = b.thread_id_x(); row = b.block_id_x() */
    tid = rocke_b_thread_id_x(b);
    row = rocke_b_block_id_x(b);

    /* --- CK Tile data abstractions --- */
    /* x_view = make_naive_tensor_view_packed(X, shape=(1,N), dtype=io_ty) */
    shape2[0] = 1;
    shape2[1] = N;
    rocke_make_naive_tensor_view_packed(&x_view, X, shape2, 2, io_ty);
    rocke_make_naive_tensor_view_packed(&y_view, Y, shape2, 2, io_ty);
    /* g_view = make_global_view(Gamma, shape=(N,), dtype=io_ty) */
    shape1[0] = N;
    rocke_make_global_view(&g_view, Gamma, shape1, 1, io_ty, NULL);
    rocke_make_global_view(&b_view, Beta, shape1, 1, io_ty, NULL);

    /* x_tile = make_tile_window(x_view, lengths=(1,N), origin=(row, 0)) */
    lengths2[0] = 1;
    lengths2[1] = N;
    origin[0] = row;
    origin[1] = rocke_b_const_i32(b, 0);
    rocke_make_tile_window(&x_tile, &x_view, lengths2, origin, 2);
    /* y_tile = make_tile_window(y_view, lengths=(1,N), origin=(row, 0)) */
    origin[0] = row;
    origin[1] = rocke_b_const_i32(b, 0);
    rocke_make_tile_window(&y_tile, &y_view, lengths2, origin, 2);

    /* --- LDS scratch (three f32 channels of BS words each) --- */
    ldsshape[0] = BS;
    rocke_make_lds_view(b, &lds_mean_v, rocke_f32(), ldsshape, 1, "lds_mean", NULL);
    rocke_make_lds_view(b, &lds_m2_v, rocke_f32(), ldsshape, 1, "lds_m2", NULL);
    rocke_make_lds_view(b, &lds_count_v, rocke_f32(), ldsshape, 1, "lds_count", NULL);
    lds_mean = lds_mean_v.base;
    lds_m2 = lds_m2_v.base;
    lds_count = lds_count_v.base;

    /* two_pass = row_norm_needs_two_pass(elems) */
    two_pass = rocke_row_norm_needs_two_pass(elems, ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD);

    /* sum_p = b.const_f32(0.0); sumsq_p = b.const_f32(0.0) */
    p1.sum_p = rocke_b_const_f32(b, 0.0);
    p1.sumsq_p = rocke_b_const_f32(b, 0.0);

    /* Pass 1: sweep_row_chunks(b, x_tile, tid, BS, VEC, elems,
     *                          body=pass1_body, cache=not two_pass) */
    sweep_res = rocke_sweep_row_chunks(b,
                                       &x_tile,
                                       tid,
                                       BS,
                                       VEC,
                                       elems,
                                       NULL, /* row=None */
                                       ln_pass1_body,
                                       &p1,
                                       !two_pass); /* cache */

    /* --- per-thread Welford triple from sum_p / sumsq_p --- */
    mean = NULL;
    var = NULL;
    if(spec->cheap_reduce)
    {
        /* MIOpen-style cheap reduction: mean, var = welford_block_reduce(b,
         *     sum_p, sumsq_p, elems, lds_mean, lds_m2, tid, block_size=BS) */
        if(spec->fast_reduce)
        {
            rocke_welford_block_reduce_wave_prologue(b, p1.sum_p, p1.sumsq_p, elems, lds_mean,
                                                     lds_m2, tid, BS, spec->wave_size, &mean,
                                                     &var);
        }
        else
        {
            rocke_welford_block_reduce(
                b, p1.sum_p, p1.sumsq_p, elems, lds_mean, lds_m2, tid, BS, &mean, &var);
        }
    }
    else
    {
        /* count_p = float(elems); inv_count_p = b.const_f32(1.0 / count_p) */
        count_p = (double)elems;
        inv_count_p = rocke_b_const_f32(b, 1.0 / count_p);
        /* mean_p = b.fmul(sum_p, inv_count_p) */
        mean_p = rocke_b_fmul(b, p1.sum_p, inv_count_p);
        /* m2_p = b.fsub(sumsq_p, b.fmul(mean_p, sum_p)) */
        m2_p = rocke_b_fsub(b, p1.sumsq_p, rocke_b_fmul(b, mean_p, p1.sum_p));

        /* mean, var = welford_block_reduce_stable(b, mean_p, m2_p,
         *     b.const_f32(count_p), lds_mean, lds_m2, lds_count, tid,
         *     block_size=BS) */
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
    }

    /* inv_std = b.rsqrt(b.fadd(var, eps)) */
    inv_std = rocke_b_rsqrt(b, rocke_b_fadd(b, var, eps));

    /* if save_mean_invstd:
     *     with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
     *         store_scalar_from_f32(b, Mean,   row, mean,    dtype)
     *         store_scalar_from_f32(b, InvStd, row, inv_std, dtype) */
    if(spec->save_mean_invstd)
    {
        rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
        rocke_b_region_enter(b, iff.then_region);
        rocke_b_store_scalar_from_f32(b, Mean, row, mean, spec->dtype);
        rocke_b_store_scalar_from_f32(b, InvStd, row, inv_std, spec->dtype);
        rocke_b_region_leave(b);
    }

    /* Pass 2: pass2_row_chunks(b, y_tile, tid, BS, VEC, elems,
     *                          body=pass2_body, cached_f32=sweep_res.cached) */
    p2.two_pass = two_pass;
    p2.vec = VEC;
    p2.x_tile = &x_tile;
    p2.g_view = &g_view;
    p2.b_view = &b_view;
    p2.mean = mean;
    p2.inv_std = inv_std;

    rocke_pass2_row_chunks(b,
                           &y_tile,
                           tid,
                           BS,
                           VEC,
                           elems,
                           NULL, /* row=None */
                           ln_pass2_body,
                           &p2,
                           sweep_res.cached,
                           sweep_res.num_cached);

    /* return b.kernel */
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_layernorm2d_new(rocke_ir_builder_t* b,
                                                const rocke_layernorm2d_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_layernorm2d_kernel_name(spec, name, sizeof name) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_layernorm2d(b, spec);
    });
}

/* ------------------------------------------------------------------ *
 * grid / signature
 * ------------------------------------------------------------------ */

rocke_status_t rocke_layernorm2d_grid(int m, const rocke_layernorm2d_spec_t* spec, int out[3])
{
    int totals[1];
    int tiles[1];
    (void)spec;
    if(out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* ceil_div_grid((m, 1)) -- one (total, tile) pair: (m, 1). */
    totals[0] = m;
    tiles[0] = 1;
    return rocke_ceil_div_grid(totals, tiles, 1, out);
}

/* layernorm2d_signature(spec):
 *   SignatureBuilder().ptr("X",dt).ptr("Gamma",dt).ptr("Beta",dt).ptr("Y",dt)
 *   [.ptr("Mean",dt).ptr("InvStd",dt)]
 *   .scalar("M","i32").scalar("N","i32").scalar("eps","f32").build() */
rocke_status_t rocke_layernorm2d_signature(rocke_signature_builder_t* sb,
                                           const rocke_layernorm2d_spec_t* spec,
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
    if(spec->save_mean_invstd)
    {
        rocke_signature_builder_ptr(sb, "Mean", spec->dtype, NULL);
        rocke_signature_builder_ptr(sb, "InvStd", spec->dtype, NULL);
    }
    rocke_signature_builder_scalar(sb, "M", "i32");
    rocke_signature_builder_scalar(sb, "N", "i32");
    rocke_signature_builder_scalar(sb, "eps", "f32");
    return rocke_signature_builder_build(sb, out_items, out_count);
}

/* ------------------------------------------------------------------ *
 * lower-to-.ll convenience
 * ------------------------------------------------------------------ */

rocke_status_t rocke_layernorm2d_lower_to_llvm(const rocke_layernorm2d_spec_t* spec,
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
            ln_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_layernorm2d_new(&b, spec);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            ln_set_reason(err, err_cap, m != NULL ? m : "build_layernorm2d failed");
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
