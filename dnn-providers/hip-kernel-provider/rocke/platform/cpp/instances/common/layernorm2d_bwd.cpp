// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_layernorm2d_bwd.c -- C99 port of
 * rocke/instances/common/layernorm2d_bwd.py.
 *
 * Byte-faithful reproduction of build_layernorm2d_bwd()'s builder-call
 * sequence. See instance_layernorm2d_bwd.h for the public surface.
 */

#include "rocke/instance_layernorm2d_bwd.h"

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
 * spec defaults / name
 * ------------------------------------------------------------------ */

rocke_layernorm2d_bwd_spec_t rocke_layernorm2d_bwd_spec_default(void)
{
    rocke_layernorm2d_bwd_spec_t s;
    s.part = "dx";
    s.block_size = 256;
    s.vec = 4;
    s.dtype = "f16";
    s.wave_size = 64;
    s.fast_reduce = false;
    s.unroll = 4;
    s.name = "rocke_layernorm2d_bwd";
    return s;
}

static bool lnb_is_part(const rocke_layernorm2d_bwd_spec_t* spec, const char* part)
{
    return spec->part != NULL && strcmp(spec->part, part) == 0;
}

/* LayerNorm2DBwdSpec.kernel_name():
 *   kernel_name_join(f"{name}_{part}", dtype, f"b{block_size}", f"v{vec}",
 *                    f"u{unroll}" if part != "dx" else "",
 *                    flags={"fr": fast_reduce})
 * (kernel_name_join drops the empty part.) */
rocke_status_t rocke_layernorm2d_bwd_kernel_name(const rocke_layernorm2d_bwd_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char prefix[128];
    char part_b[32];
    char part_v[32];
    char part_u[32];
    const char* parts[4];
    const char* flag_names[1];
    int flag_on[1];
    int nparts = 0;

    if(spec == NULL || out == NULL || spec->name == NULL || spec->part == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    snprintf(prefix, sizeof prefix, "%s_%s", spec->name, spec->part);
    snprintf(part_b, sizeof part_b, "b%d", spec->block_size);
    snprintf(part_v, sizeof part_v, "v%d", spec->vec);
    snprintf(part_u, sizeof part_u, "u%d", spec->unroll);

    parts[nparts++] = spec->dtype;
    parts[nparts++] = part_b;
    parts[nparts++] = part_v;
    if(!lnb_is_part(spec, "dx"))
    {
        parts[nparts++] = part_u;
    }

    flag_names[0] = "fr";
    flag_on[0] = spec->fast_reduce ? 1 : 0;

    return rocke_kernel_name_join(prefix, parts, nparts, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ */

static void lnb_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

bool rocke_layernorm2d_bwd_is_valid_spec(const rocke_layernorm2d_bwd_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    const rocke_archtarget_t* target;
    rocke_io_spec_rule_t rule;
    rocke_arena_t arena;
    const char* why = NULL;
    int io_ok;

    if(reason != NULL && reason_cap > 0)
    {
        reason[0] = '\0';
    }
    if(spec == NULL)
    {
        lnb_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        lnb_set_reason(reason, reason_cap, "unknown arch");
        return false;
    }

    if(!lnb_is_part(spec, "dx") && !lnb_is_part(spec, "dwdb") && !lnb_is_part(spec, "dwdb_reduce"))
    {
        lnb_set_reason(reason, reason_cap, "unknown part");
        return false;
    }

    rocke_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec);
    rule.n_per_block_set = 0;
    rule.max_elems_per_thread_set = 0;
    if(rocke_arena_init(&arena, 4096) != ROCKE_OK)
    {
        lnb_set_reason(reason, reason_cap, "arena init failed");
        return false;
    }
    io_ok = rocke_validate_io(&arena, &rule, &why);
    if(!io_ok)
    {
        lnb_set_reason(reason, reason_cap, why != NULL ? why : "validate_io failed");
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    if(spec->block_size > rocke_archtarget_max_threads_per_block(target))
    {
        lnb_set_reason(reason, reason_cap, "block_size > max_threads_per_block");
        return false;
    }

    if(spec->unroll != 1 && spec->unroll != 2 && spec->unroll != 4 && spec->unroll != 8)
    {
        lnb_set_reason(reason, reason_cap, "unroll not in (1, 2, 4, 8)");
        return false;
    }

    if(lnb_is_part(spec, "dx"))
    {
        if(spec->unroll != 4)
        {
            lnb_set_reason(reason, reason_cap, "unroll only applies to the dwdb / dwdb_reduce parts");
            return false;
        }
        if(spec->fast_reduce && spec->block_size % spec->wave_size != 0)
        {
            lnb_set_reason(
                reason, reason_cap, "fast_reduce requires block_size divisible by wave_size");
            return false;
        }
        /* Two f32 twin-reduction buffers: 2 * block_size * 4 bytes. */
        if(!rocke_archtarget_fits_lds(target, (long)2 * (long)spec->block_size * 4L))
        {
            lnb_set_reason(reason, reason_cap, "LDS budget exceeds cap");
            return false;
        }
    }
    else if(spec->fast_reduce)
    {
        lnb_set_reason(reason, reason_cap, "fast_reduce only applies to the dx part");
        return false;
    }

    return true;
}

/* ------------------------------------------------------------------ *
 * shared emission helpers
 * ------------------------------------------------------------------ */

static rocke_value_t*
    lnb_fadd_combine(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, void* user)
{
    (void)user;
    return rocke_b_fadd(b, a, c);
}

static rocke_value_t* lnb_param_ro(rocke_ir_builder_t* b,
                                   const char* name,
                                   const rocke_type_t* elem,
                                   bool aligned)
{
    rocke_param_opts_t opts;
    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    if(aligned)
    {
        opts.align = 16;
        opts.align_set = true;
    }
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &opts);
}

static rocke_value_t* lnb_param_wo(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem)
{
    rocke_param_opts_t opts;
    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &opts);
}

/* ------------------------------------------------------------------ *
 * part "dx"
 * ------------------------------------------------------------------ */

typedef struct lnb_dx_ctx
{
    const rocke_tile_window_t* dy_tile;
    const rocke_tile_window_t* x_tile;
    const rocke_tensor_view_t* g_view;
    rocke_value_t* rstd;
    rocke_value_t* coef_a;
    rocke_value_t* coef_b;
} lnb_dx_ctx_t;

/* dx_of(dy, g, x):
 *   b.fsub(b.fsub(b.fmul(rstd, b.fmul(dy, g)), b.fmul(coef_a, x)), coef_b) */
static rocke_value_t* lnb_dx_of(rocke_ir_builder_t* b,
                                const lnb_dx_ctx_t* ctx,
                                rocke_value_t* dy,
                                rocke_value_t* g,
                                rocke_value_t* x)
{
    rocke_value_t* t = rocke_b_fmul(b, ctx->rstd, rocke_b_fmul(b, dy, g));
    rocke_value_t* ax = rocke_b_fmul(b, ctx->coef_a, x);
    return rocke_b_fsub(b, rocke_b_fsub(b, t, ax), ctx->coef_b);
}

/* pass1_body(n_off, dy_scalars, iter_vars):
 *   x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
 *   gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
 *   dyg  = [b.fmul(dy_scalars[i], gv[i]) for i]
 *   dygx = [b.fmul(dyg[i], x_scalars[i]) for i]
 *   s_dyg  = b.fadd(s_dyg,  tree_reduce(b, b.fadd, dyg))
 *   s_dygx = b.fadd(s_dygx, tree_reduce(b, b.fadd, dygx)) */
static void lnb_dx_pass1_body(rocke_ir_builder_t* b,
                              rocke_value_t* n_off,
                              rocke_value_t* const* dy_scalars,
                              int vec,
                              rocke_value_t* const* iter_vars,
                              int num_iter_vars,
                              rocke_value_t** new_iter_vals,
                              void* user)
{
    lnb_dx_ctx_t* ctx = (lnb_dx_ctx_t*)user;
    rocke_value_t* xs[8];
    rocke_value_t* gv[8];
    rocke_value_t* dyg[8];
    rocke_value_t* dygx[8];
    rocke_value_t* li[2];
    rocke_value_t* idx1[1];
    int i;
    (void)num_iter_vars;

    li[0] = rocke_b_const_i32(b, 0);
    li[1] = n_off;
    rocke_tile_window_load_vec_as_f32(b, ctx->x_tile, li, 2, vec, xs);
    idx1[0] = n_off;
    rocke_tensor_view_load_vec_as_f32(b, ctx->g_view, idx1, 1, vec, gv);
    for(i = 0; i < vec; ++i)
    {
        dyg[i] = rocke_b_fmul(b, dy_scalars[i], gv[i]);
    }
    for(i = 0; i < vec; ++i)
    {
        dygx[i] = rocke_b_fmul(b, dyg[i], xs[i]);
    }
    new_iter_vals[0]
        = rocke_b_fadd(b, iter_vars[0], rocke_tree_reduce(b, lnb_fadd_combine, NULL, dyg, vec));
    new_iter_vals[1]
        = rocke_b_fadd(b, iter_vars[1], rocke_tree_reduce(b, lnb_fadd_combine, NULL, dygx, vec));
}

/* pass2_body(n_off):
 *   dys = dy_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
 *   xs  = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
 *   gv  = g_view.load_vec_as_f32(b, [n_off], n=VEC)
 *   return [dx_of(dys[i], gv[i], xs[i]) for i] */
static void lnb_dx_pass2_body(
    rocke_ir_builder_t* b, rocke_value_t* n_off, rocke_value_t** out, int vec, void* user)
{
    lnb_dx_ctx_t* ctx = (lnb_dx_ctx_t*)user;
    rocke_value_t* dys[8];
    rocke_value_t* xs[8];
    rocke_value_t* gv[8];
    rocke_value_t* li[2];
    rocke_value_t* idx1[1];
    int i;

    li[0] = rocke_b_const_i32(b, 0);
    li[1] = n_off;
    rocke_tile_window_load_vec_as_f32(b, ctx->dy_tile, li, 2, vec, dys);
    li[0] = rocke_b_const_i32(b, 0);
    li[1] = n_off;
    rocke_tile_window_load_vec_as_f32(b, ctx->x_tile, li, 2, vec, xs);
    idx1[0] = n_off;
    rocke_tensor_view_load_vec_as_f32(b, ctx->g_view, idx1, 1, vec, gv);
    for(i = 0; i < vec; ++i)
    {
        out[i] = lnb_dx_of(b, ctx, dys[i], gv[i], xs[i]);
    }
}

static rocke_kernel_def_t* lnb_build_dx(rocke_ir_builder_t* b,
                                        const rocke_layernorm2d_bwd_spec_t* spec)
{
    const rocke_type_t* io_ty;
    int BS, VEC, CAP, tc;
    rocke_value_t *DY, *X, *Gamma, *Mean, *Rstd, *DX, *cols;
    rocke_value_t *tid, *row, *mean, *rstd;
    rocke_tensor_view_t dy_view, x_view, dx_view, g_view, lds_a_v, lds_c_v;
    rocke_tile_window_t dy_tile, x_tile, dx_tile;
    rocke_value_t *lds_a, *lds_c;
    rocke_stride_t row_stride[2];
    int shape2[2];
    int shape1[1];
    int ldsshape[1];
    int lengths2[2];
    rocke_value_t* origin[2];
    rocke_param_opts_t opts;
    rocke_iter_arg_t iter_args[2];
    rocke_sweep_dynamic_result_t sweep_res;
    rocke_value_t *s_dyg, *s_dygx, *main_elems, *t_dyg = NULL, *t_dygx = NULL;
    rocke_value_t *inv_n, *rstd3, *trip_count;
    lnb_dx_ctx_t ctx;

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
        return NULL;
    BS = spec->block_size;
    VEC = spec->vec;
    CAP = BS * VEC;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    DY = lnb_param_ro(b, "DY", io_ty, true);
    X = lnb_param_ro(b, "X", io_ty, true);
    Gamma = lnb_param_ro(b, "Gamma", io_ty, true);
    Mean = lnb_param_ro(b, "Mean", io_ty, false);
    Rstd = lnb_param_ro(b, "Rstd", io_ty, false);
    DX = lnb_param_wo(b, "DX", io_ty);
    (void)rocke_b_param(b, "M", rocke_i32(), NULL);
    cols = rocke_b_param(b, "N", rocke_i32(), NULL);
    (void)opts;

    tid = rocke_b_thread_id_x(b);
    row = rocke_b_block_id_x(b);

    mean = rocke_b_load_scalar_as_f32(b, Mean, row, spec->dtype);
    rstd = rocke_b_load_scalar_as_f32(b, Rstd, row, spec->dtype);

    shape2[0] = 1;
    shape2[1] = CAP;
    row_stride[0] = rocke_stride_value(cols);
    row_stride[1] = rocke_stride_imm(1);
    rocke_make_global_view(&dy_view, DY, shape2, 2, io_ty, row_stride);
    rocke_make_global_view(&x_view, X, shape2, 2, io_ty, row_stride);
    rocke_make_global_view(&dx_view, DX, shape2, 2, io_ty, row_stride);
    shape1[0] = CAP;
    rocke_make_global_view(&g_view, Gamma, shape1, 1, io_ty, NULL);

    lengths2[0] = 1;
    lengths2[1] = 1;
    origin[0] = row;
    origin[1] = rocke_b_const_i32(b, 0);
    rocke_make_tile_window(&dy_tile, &dy_view, lengths2, origin, 2);
    origin[1] = rocke_b_const_i32(b, 0);
    rocke_make_tile_window(&x_tile, &x_view, lengths2, origin, 2);
    origin[1] = rocke_b_const_i32(b, 0);
    rocke_make_tile_window(&dx_tile, &dx_view, lengths2, origin, 2);

    ldsshape[0] = BS;
    rocke_make_lds_view(b, &lds_a_v, rocke_f32(), ldsshape, 1, "lds_dyg", NULL);
    rocke_make_lds_view(b, &lds_c_v, rocke_f32(), ldsshape, 1, "lds_dygx", NULL);
    lds_a = lds_a_v.base;
    lds_c = lds_c_v.base;

    memset(&ctx, 0, sizeof ctx);
    ctx.dy_tile = &dy_tile;
    ctx.x_tile = &x_tile;
    ctx.g_view = &g_view;

    /* --- Pass 1: sum(dY*gamma), sum(dY*gamma*X) --- */
    iter_args[0].name = "s_dyg";
    iter_args[0].init = rocke_b_const_f32(b, 0.0);
    iter_args[1].name = "s_dygx";
    iter_args[1].init = rocke_b_const_f32(b, 0.0);
    sweep_res = rocke_sweep_row_chunks_dynamic(
        b, &dy_tile, tid, BS, VEC, cols, iter_args, 2, NULL, lnb_dx_pass1_body, &ctx, "k0");
    s_dyg = sweep_res.final_iter_vals[0];
    s_dygx = sweep_res.final_iter_vals[1];
    main_elems = sweep_res.main_elems_per_row;

    for(tc = 0; tc < VEC; ++tc)
    {
        rocke_value_t* idx_col = rocke_b_add(
            b, rocke_b_add(b, main_elems, rocke_b_const_i32(b, tc * BS)), tid);
        rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, cols);
        rocke_value_t* idx_safe
            = rocke_b_smin(b, idx_col, rocke_b_sub(b, cols, rocke_b_const_i32(b, 1)));
        rocke_value_t* flat = rocke_b_add(b, rocke_b_mul(b, row, cols), idx_safe);
        rocke_value_t* dy_raw = rocke_b_load_scalar_as_f32(b, DY, flat, spec->dtype);
        rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
        rocke_value_t* g_raw = rocke_b_load_scalar_as_f32(b, Gamma, idx_safe, spec->dtype);
        rocke_value_t* prod = rocke_b_fmul(b, dy_raw, g_raw);
        rocke_value_t* dyg = rocke_b_select(b, valid, prod, rocke_b_const_f32(b, 0.0));
        s_dyg = rocke_b_fadd(b, s_dyg, dyg);
        s_dygx = rocke_b_fadd(b, s_dygx, rocke_b_fmul(b, dyg, x_raw));
    }

    if(spec->fast_reduce)
    {
        rocke_block_lds_reduce_pair_with_wave_prologue(b, s_dyg, s_dygx, lds_a, lds_c, tid, BS,
                                                       ROCKE_REDUCE_SUM, ROCKE_REDUCE_SUM,
                                                       spec->wave_size, &t_dyg, &t_dygx);
    }
    else
    {
        rocke_block_lds_reduce_pair(b, s_dyg, s_dygx, lds_a, lds_c, tid, BS, ROCKE_REDUCE_SUM,
                                    ROCKE_REDUCE_SUM, &t_dyg, &t_dygx);
    }
    if(!rocke_ir_builder_ok(b))
        return NULL;

    /* inv_n = b.fdiv(b.const_f32(1.0), b.sitofp_f32(cols)) */
    {
        rocke_value_t* one_f = rocke_b_const_f32(b, 1.0);
        inv_n = rocke_b_fdiv(b, one_f, rocke_b_sitofp_f32(b, cols));
    }
    rstd3 = rocke_b_fmul(b, rocke_b_fmul(b, rstd, rstd), rstd);
    /* coef_a = fmul(fmul(rstd3, inv_n), fsub(t_dygx, fmul(t_dyg, mean))) */
    {
        rocke_value_t* r3n = rocke_b_fmul(b, rstd3, inv_n);
        rocke_value_t* d = rocke_b_fsub(b, t_dygx, rocke_b_fmul(b, t_dyg, mean));
        ctx.coef_a = rocke_b_fmul(b, r3n, d);
    }
    /* coef_b = fsub(fmul(fmul(rstd, t_dyg), inv_n), fmul(coef_a, mean)) */
    {
        rocke_value_t* rn = rocke_b_fmul(b, rocke_b_fmul(b, rstd, t_dyg), inv_n);
        ctx.coef_b = rocke_b_fsub(b, rn, rocke_b_fmul(b, ctx.coef_a, mean));
    }
    ctx.rstd = rstd;

    /* --- Pass 2: dX --- */
    trip_count = rocke_b_div(b, cols, rocke_b_const_i32(b, BS * VEC));
    rocke_pass2_row_chunks_dynamic(
        b, &dx_tile, tid, BS, VEC, main_elems, trip_count, NULL, lnb_dx_pass2_body, &ctx, "k0");

    for(tc = 0; tc < VEC; ++tc)
    {
        rocke_value_t* idx_col = rocke_b_add(
            b, rocke_b_add(b, main_elems, rocke_b_const_i32(b, tc * BS)), tid);
        rocke_value_t* valid = rocke_b_cmp_lt(b, idx_col, cols);
        rocke_if_t iff = rocke_b_scf_if(b, valid);
        rocke_b_region_enter(b, iff.then_region);
        {
            rocke_value_t* flat = rocke_b_add(b, rocke_b_mul(b, row, cols), idx_col);
            rocke_value_t* dy_raw = rocke_b_load_scalar_as_f32(b, DY, flat, spec->dtype);
            rocke_value_t* x_raw = rocke_b_load_scalar_as_f32(b, X, flat, spec->dtype);
            rocke_value_t* g_raw = rocke_b_load_scalar_as_f32(b, Gamma, idx_col, spec->dtype);
            rocke_b_store_scalar_from_f32(
                b, DX, flat, lnb_dx_of(b, &ctx, dy_raw, g_raw, x_raw), spec->dtype);
        }
        rocke_b_region_leave(b);
    }

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

/* ------------------------------------------------------------------ *
 * part "dwdb"
 * ------------------------------------------------------------------ */

typedef struct lnb_row
{
    rocke_value_t* dys[8];
    rocke_value_t* xs[8];
    rocke_value_t* mean;
    rocke_value_t* rstd;
} lnb_row_t;

/* load_row(r):
 *   flat = b.add(b.mul(r, cols), c_safe)
 *   dys = load_vec_as_f32(b, DY, flat, dtype, n=VEC)
 *   xs  = load_vec_as_f32(b, X,  flat, dtype, n=VEC)
 *   mean = load_scalar_as_f32(b, Mean, r, dtype)
 *   rstd = load_scalar_as_f32(b, Rstd, r, dtype) */
static void lnb_load_row(rocke_ir_builder_t* b,
                         const rocke_layernorm2d_bwd_spec_t* spec,
                         rocke_value_t* DY,
                         rocke_value_t* X,
                         rocke_value_t* Mean,
                         rocke_value_t* Rstd,
                         rocke_value_t* cols,
                         rocke_value_t* c_safe,
                         rocke_value_t* r,
                         lnb_row_t* out)
{
    rocke_value_t* flat = rocke_b_add(b, rocke_b_mul(b, r, cols), c_safe);
    rocke_b_load_vec_as_f32(b, DY, flat, spec->dtype, spec->vec, out->dys);
    rocke_b_load_vec_as_f32(b, X, flat, spec->dtype, spec->vec, out->xs);
    out->mean = rocke_b_load_scalar_as_f32(b, Mean, r, spec->dtype);
    out->rstd = rocke_b_load_scalar_as_f32(b, Rstd, r, spec->dtype);
}

/* _dwdb_accumulate: for i: xhat = fmul(fsub(xs[i], mean), rstd);
 *   accs[i] = fma(dys[i], xhat, accs[i]); accs[vec+i] = fadd(accs[vec+i], dys[i]) */
static void lnb_dwdb_accumulate(rocke_ir_builder_t* b, rocke_value_t** accs, const lnb_row_t* r, int vec)
{
    int i;
    for(i = 0; i < vec; ++i)
    {
        rocke_value_t* xhat = rocke_b_fmul(b, rocke_b_fsub(b, r->xs[i], r->mean), r->rstd);
        accs[i] = rocke_b_fma(b, r->dys[i], xhat, accs[i]);
        accs[vec + i] = rocke_b_fadd(b, accs[vec + i], r->dys[i]);
    }
}

static rocke_kernel_def_t* lnb_build_dwdb(rocke_ir_builder_t* b,
                                          const rocke_layernorm2d_bwd_spec_t* spec)
{
    const rocke_type_t* io_ty;
    int BS, VEC, U, i, u;
    rocke_value_t *DY, *X, *Mean, *Rstd, *DW, *DB, *WS, *rows, *cols, *splits;
    rocke_value_t *tid, *bx, *by, *c_one, *c0, *valid, *c_safe;
    rocke_value_t *rows_per_split, *r0, *r1, *n_rows, *trip, *rem_lo;
    rocke_iter_arg_t iter_args[16];
    char names[16][8];
    char rnames[16][8];
    rocke_value_t* accs[16];
    lnb_row_t loaded[8];
    rocke_for_t main_f, rem_f;

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
        return NULL;
    BS = spec->block_size;
    VEC = spec->vec;
    U = spec->unroll;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    DY = lnb_param_ro(b, "DY", io_ty, true);
    X = lnb_param_ro(b, "X", io_ty, true);
    Mean = lnb_param_ro(b, "Mean", io_ty, false);
    Rstd = lnb_param_ro(b, "Rstd", io_ty, false);
    DW = lnb_param_wo(b, "DW", io_ty);
    DB = lnb_param_wo(b, "DB", io_ty);
    WS = lnb_param_wo(b, "WS", rocke_f32());
    rows = rocke_b_param(b, "M", rocke_i32(), NULL);
    cols = rocke_b_param(b, "N", rocke_i32(), NULL);
    splits = rocke_b_param(b, "P", rocke_i32(), NULL);

    tid = rocke_b_thread_id_x(b);
    bx = rocke_b_block_id_x(b);
    by = rocke_b_block_id_y(b);

    c_one = rocke_b_const_i32(b, 1);
    {
        rocke_value_t* t = rocke_b_add(b, rocke_b_mul(b, bx, rocke_b_const_i32(b, BS)), tid);
        c0 = rocke_b_mul(b, t, rocke_b_const_i32(b, VEC));
    }
    valid = rocke_b_cmp_lt(b, c0, cols);
    c_safe = rocke_b_select(b, valid, c0, rocke_b_const_i32(b, 0));

    rows_per_split = rocke_b_div(b, rocke_b_add(b, rows, rocke_b_sub(b, splits, c_one)), splits);
    r0 = rocke_b_mul(b, by, rows_per_split);
    r1 = rocke_b_smin(b, rocke_b_add(b, r0, rows_per_split), rows);
    {
        rocke_value_t* d = rocke_b_sub(b, r1, r0);
        n_rows = rocke_b_smax(b, d, rocke_b_const_i32(b, 0));
    }
    trip = rocke_b_div(b, n_rows, rocke_b_const_i32(b, U));

    for(i = 0; i < VEC; ++i)
    {
        snprintf(names[i], sizeof names[i], "dw%d", i);
        iter_args[i].name = names[i];
        iter_args[i].init = rocke_b_const_f32(b, 0.0);
    }
    for(i = 0; i < VEC; ++i)
    {
        snprintf(names[VEC + i], sizeof names[VEC + i], "db%d", i);
        iter_args[VEC + i].name = names[VEC + i];
        iter_args[VEC + i].init = rocke_b_const_f32(b, 0.0);
    }
    main_f = rocke_b_scf_for_iter(
        b, rocke_b_const_i32(b, 0), trip, c_one, iter_args, 2 * VEC, "rk", false, true);
    rocke_b_region_enter(b, main_f.body);
    {
        rocke_value_t* r_base = rocke_b_add(b, r0, rocke_b_mul(b, main_f.iv, rocke_b_const_i32(b, U)));
        for(u = 0; u < U; ++u)
        {
            rocke_value_t* r = rocke_b_add(b, r_base, rocke_b_const_i32(b, u));
            lnb_load_row(b, spec, DY, X, Mean, Rstd, cols, c_safe, r, &loaded[u]);
        }
        for(i = 0; i < 2 * VEC; ++i)
            accs[i] = main_f.iter_vars[i];
        for(u = 0; u < U; ++u)
            lnb_dwdb_accumulate(b, accs, &loaded[u], VEC);
        rocke_b_scf_yield(b, accs, 2 * VEC);
    }
    rocke_b_region_leave(b);

    rem_lo = rocke_b_add(b, r0, rocke_b_mul(b, trip, rocke_b_const_i32(b, U)));
    for(i = 0; i < VEC; ++i)
    {
        snprintf(rnames[i], sizeof rnames[i], "rdw%d", i);
        iter_args[i].name = rnames[i];
        iter_args[i].init = main_f.op->results[i];
    }
    for(i = 0; i < VEC; ++i)
    {
        snprintf(rnames[VEC + i], sizeof rnames[VEC + i], "rdb%d", i);
        iter_args[VEC + i].name = rnames[VEC + i];
        iter_args[VEC + i].init = main_f.op->results[VEC + i];
    }
    rem_f = rocke_b_scf_for_iter(b, rem_lo, r1, c_one, iter_args, 2 * VEC, "rr", false, true);
    rocke_b_region_enter(b, rem_f.body);
    {
        lnb_load_row(b, spec, DY, X, Mean, Rstd, cols, c_safe, rem_f.iv, &loaded[0]);
        for(i = 0; i < 2 * VEC; ++i)
            accs[i] = rem_f.iter_vars[i];
        lnb_dwdb_accumulate(b, accs, &loaded[0], VEC);
        rocke_b_scf_yield(b, accs, 2 * VEC);
    }
    rocke_b_region_leave(b);

    {
        rocke_value_t* const* dw = rem_f.op->results;
        rocke_value_t* const* db = rem_f.op->results + VEC;
        rocke_if_t iff = rocke_b_scf_if(b, valid);
        rocke_b_region_enter(b, iff.then_region);
        {
            rocke_if_else_t ie = rocke_b_scf_if_else(b, rocke_b_cmp_eq(b, splits, c_one));
            rocke_b_region_enter(b, ie.then_region);
            rocke_b_store_vec(b, DW, c0, rocke_b_pack_f32_to(b, dw, VEC, spec->dtype), VEC);
            rocke_b_store_vec(b, DB, c0, rocke_b_pack_f32_to(b, db, VEC, spec->dtype), VEC);
            rocke_b_region_leave(b);
            rocke_b_region_enter(b, ie.else_region);
            {
                rocke_value_t* ws_base
                    = rocke_b_mul(b, rocke_b_mul(b, by, rocke_b_const_i32(b, 2)), cols);
                rocke_value_t* i_dw = rocke_b_add(b, ws_base, c0);
                rocke_value_t* v_dw = rocke_b_vec_pack(b, dw, VEC, rocke_f32());
                rocke_b_global_store_vN(b, WS, i_dw, v_dw, VEC, 0);
                rocke_value_t* i_db = rocke_b_add(b, rocke_b_add(b, ws_base, cols), c0);
                rocke_value_t* v_db = rocke_b_vec_pack(b, db, VEC, rocke_f32());
                rocke_b_global_store_vN(b, WS, i_db, v_db, VEC, 0);
            }
            rocke_b_region_leave(b);
        }
        rocke_b_region_leave(b);
    }

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

/* ------------------------------------------------------------------ *
 * part "dwdb_reduce"
 * ------------------------------------------------------------------ */

/* load_partial(p): v = b.global_load_vN(WS, b.add(b.mul(p, width), e_safe), F32, VEC);
 *                  return [b.vec_extract(v, i) for i] */
static void lnb_load_partial(rocke_ir_builder_t* b,
                             rocke_value_t* WS,
                             rocke_value_t* width,
                             rocke_value_t* e_safe,
                             rocke_value_t* p,
                             int vec,
                             rocke_value_t** out)
{
    rocke_value_t* idx = rocke_b_add(b, rocke_b_mul(b, p, width), e_safe);
    rocke_value_t* v = rocke_b_global_load_vN(b, WS, idx, rocke_f32(), vec, 0);
    int i;
    for(i = 0; i < vec; ++i)
        out[i] = rocke_b_vec_extract(b, v, i);
}

static rocke_kernel_def_t* lnb_build_dwdb_reduce(rocke_ir_builder_t* b,
                                                 const rocke_layernorm2d_bwd_spec_t* spec)
{
    const rocke_type_t* io_ty;
    int BS, VEC, U, i, u;
    rocke_value_t *WS, *DW, *DB, *cols, *splits;
    rocke_value_t *tid, *bx, *c_one, *width, *e0, *valid, *e_safe, *trip, *rem_lo;
    rocke_iter_arg_t iter_args[8];
    char names[8][8];
    char rnames[8][8];
    rocke_value_t* accs[8];
    rocke_value_t* loaded[8][8];
    rocke_for_t main_f, rem_f;

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
        return NULL;
    BS = spec->block_size;
    VEC = spec->vec;
    U = spec->unroll;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    WS = lnb_param_ro(b, "WS", rocke_f32(), true);
    DW = lnb_param_wo(b, "DW", io_ty);
    DB = lnb_param_wo(b, "DB", io_ty);
    cols = rocke_b_param(b, "N", rocke_i32(), NULL);
    splits = rocke_b_param(b, "P", rocke_i32(), NULL);

    tid = rocke_b_thread_id_x(b);
    bx = rocke_b_block_id_x(b);

    c_one = rocke_b_const_i32(b, 1);
    width = rocke_b_mul(b, cols, rocke_b_const_i32(b, 2));
    {
        rocke_value_t* t = rocke_b_add(b, rocke_b_mul(b, bx, rocke_b_const_i32(b, BS)), tid);
        e0 = rocke_b_mul(b, t, rocke_b_const_i32(b, VEC));
    }
    valid = rocke_b_cmp_lt(b, e0, width);
    e_safe = rocke_b_select(b, valid, e0, rocke_b_const_i32(b, 0));
    trip = rocke_b_div(b, splits, rocke_b_const_i32(b, U));

    for(i = 0; i < VEC; ++i)
    {
        snprintf(names[i], sizeof names[i], "s%d", i);
        iter_args[i].name = names[i];
        iter_args[i].init = rocke_b_const_f32(b, 0.0);
    }
    main_f = rocke_b_scf_for_iter(
        b, rocke_b_const_i32(b, 0), trip, c_one, iter_args, VEC, "pk", false, true);
    rocke_b_region_enter(b, main_f.body);
    {
        rocke_value_t* p_base = rocke_b_mul(b, main_f.iv, rocke_b_const_i32(b, U));
        for(u = 0; u < U; ++u)
        {
            rocke_value_t* p = rocke_b_add(b, p_base, rocke_b_const_i32(b, u));
            lnb_load_partial(b, WS, width, e_safe, p, VEC, loaded[u]);
        }
        for(i = 0; i < VEC; ++i)
            accs[i] = main_f.iter_vars[i];
        for(u = 0; u < U; ++u)
            for(i = 0; i < VEC; ++i)
                accs[i] = rocke_b_fadd(b, accs[i], loaded[u][i]);
        rocke_b_scf_yield(b, accs, VEC);
    }
    rocke_b_region_leave(b);

    rem_lo = rocke_b_mul(b, trip, rocke_b_const_i32(b, U));
    for(i = 0; i < VEC; ++i)
    {
        snprintf(rnames[i], sizeof rnames[i], "rs%d", i);
        iter_args[i].name = rnames[i];
        iter_args[i].init = main_f.op->results[i];
    }
    rem_f = rocke_b_scf_for_iter(b, rem_lo, splits, c_one, iter_args, VEC, "pr", false, true);
    rocke_b_region_enter(b, rem_f.body);
    {
        for(i = 0; i < VEC; ++i)
            accs[i] = rem_f.iter_vars[i];
        lnb_load_partial(b, WS, width, e_safe, rem_f.iv, VEC, loaded[0]);
        for(i = 0; i < VEC; ++i)
            accs[i] = rocke_b_fadd(b, accs[i], loaded[0][i]);
        rocke_b_scf_yield(b, accs, VEC);
    }
    rocke_b_region_leave(b);

    {
        rocke_value_t* const* total = rem_f.op->results;
        rocke_if_t iff = rocke_b_scf_if(b, valid);
        rocke_b_region_enter(b, iff.then_region);
        {
            rocke_if_else_t ie = rocke_b_scf_if_else(b, rocke_b_cmp_lt(b, e0, cols));
            rocke_b_region_enter(b, ie.then_region);
            rocke_b_store_vec(b, DW, e0, rocke_b_pack_f32_to(b, total, VEC, spec->dtype), VEC);
            rocke_b_region_leave(b);
            rocke_b_region_enter(b, ie.else_region);
            {
                rocke_value_t* idx = rocke_b_sub(b, e0, cols);
                rocke_b_store_vec(
                    b, DB, idx, rocke_b_pack_f32_to(b, total, VEC, spec->dtype), VEC);
            }
            rocke_b_region_leave(b);
        }
        rocke_b_region_leave(b);
    }

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

/* ------------------------------------------------------------------ *
 * build / lower
 * ------------------------------------------------------------------ */

rocke_kernel_def_t* rocke_build_layernorm2d_bwd(rocke_ir_builder_t* b,
                                                const rocke_layernorm2d_bwd_spec_t* spec)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(!rocke_layernorm2d_bwd_is_valid_spec(spec, "gfx950", NULL, 0))
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "invalid layernorm2d_bwd spec");
        return NULL;
    }
    if(lnb_is_part(spec, "dx"))
    {
        return lnb_build_dx(b, spec);
    }
    if(lnb_is_part(spec, "dwdb"))
    {
        return lnb_build_dwdb(b, spec);
    }
    return lnb_build_dwdb_reduce(b, spec);
}

rocke_kernel_def_t* rocke_build_layernorm2d_bwd_new(rocke_ir_builder_t* b,
                                                    const rocke_layernorm2d_bwd_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_layernorm2d_bwd_kernel_name(spec, name, sizeof name) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_layernorm2d_bwd(b, spec);
    });
}

rocke_status_t rocke_layernorm2d_bwd_lower_to_llvm(const rocke_layernorm2d_bwd_spec_t* spec,
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
            lnb_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_layernorm2d_bwd_new(&b, spec);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            lnb_set_reason(err, err_cap, m != NULL ? m : "build_layernorm2d_bwd failed");
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
