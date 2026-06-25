// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_layernorm2d.c -- C99 port of
 * ck_dsl/instances/common/layernorm2d.py.
 *
 * Byte-faithful reproduction of build_layernorm2d()'s builder-call sequence.
 * See instance_layernorm2d.h for the public surface.
 */

#include "ckc/instance_layernorm2d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "ckc/helper_ck_dsl.core.arch.h"
#include "ckc/helper_ck_dsl.helpers.io.h"
#include "ckc/helper_ck_dsl.helpers.reduction.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/helper_ck_dsl.helpers.sweep.h"
#include "ckc/helper_ck_dsl.helpers.tensor_view.h"
#include "ckc/ir.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */
#include "ckc/lower_llvm.h"

/* The tensor_view f32 / lds factory peers --
 * make_naive_tensor_view_packed, make_lds_view, TensorView.load_vec_as_f32 --
 * are declared in helper_ck_dsl.helpers.tensor_view.h and defined in that
 * module's translation unit. The TileWindow.load_vec_as_f32 /
 * store_vec_from_f32 peers come from helper_ck_dsl.helpers.sweep.h. */

/* ------------------------------------------------------------------ *
 * spec defaults / properties
 * ------------------------------------------------------------------ */

ckc_layernorm2d_spec_t ckc_layernorm2d_spec_default(void)
{
    ckc_layernorm2d_spec_t s;
    s.n_per_block = 0; /* required: caller must set */
    s.block_size = 256;
    s.vec = 4;
    s.dtype = "f16";
    s.save_mean_invstd = false;
    s.wave_size = 64;
    s.name = "ck_dsl_layernorm2d_fwd";
    return s;
}

int ckc_layernorm2d_elems_per_thread(const ckc_layernorm2d_spec_t* spec)
{
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return spec->n_per_block / spec->block_size;
}

/* LayerNorm2DSpec.kernel_name():
 *   kernel_name_join(self.name, self.dtype, f"N{n_per_block}", f"b{block_size}",
 *                    f"v{vec}", flags={"smv": save_mean_invstd}) */
ckc_status_t
    ckc_layernorm2d_kernel_name(const ckc_layernorm2d_spec_t* spec, char* out, size_t out_cap)
{
    char part_n[32];
    char part_b[32];
    char part_v[32];
    const char* parts[4];
    const char* flag_names[1];
    int flag_on[1];

    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
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

    return ckc_kernel_name_join(spec->name, parts, 4, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ */

static void ln_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    ckc_spec_set_reason(reason, reason_cap, msg);
}

bool ckc_layernorm2d_is_valid_spec(const ckc_layernorm2d_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap)
{
    const ckc_archtarget_t* target;
    ckc_io_spec_rule_t rule;
    ckc_arena_t arena;
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
    target = ckc_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        ln_set_reason(reason, reason_cap, "unknown arch");
        return false;
    }

    elems = ckc_layernorm2d_elems_per_thread(spec);

    /* cap = None if row_norm_needs_two_pass(elems) else
     *       REGISTER_TILE_MAX_ELEMS_PER_THREAD */
    two_pass = ckc_row_norm_needs_two_pass(elems, CKC_REGISTER_TILE_MAX_ELEMS_PER_THREAD);

    /* validate_io(IOSpecRule(dtype, block_size, vec, n_per_block, cap)). */
    ckc_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec);
    rule.n_per_block_set = 1;
    rule.n_per_block = spec->n_per_block;
    if(two_pass)
    {
        rule.max_elems_per_thread_set = 0; /* None */
    }
    else
    {
        rule.max_elems_per_thread_set = 1;
        rule.max_elems_per_thread = CKC_REGISTER_TILE_MAX_ELEMS_PER_THREAD;
    }

    if(ckc_arena_init(&arena, 4096) != CKC_OK)
    {
        ln_set_reason(reason, reason_cap, "arena init failed");
        return false;
    }
    io_ok = ckc_validate_io(&arena, &rule, &why);
    if(!io_ok)
    {
        ln_set_reason(reason, reason_cap, why != NULL ? why : "validate_io failed");
        ckc_arena_destroy(&arena);
        return false;
    }
    ckc_arena_destroy(&arena);

    /* if block_size > target.max_threads_per_block: reject */
    max_thr = ckc_archtarget_max_threads_per_block(target);
    if(spec->block_size > max_thr)
    {
        ln_set_reason(reason, reason_cap, "block_size > max_threads_per_block");
        return false;
    }

    /* Three f32 Welford reduction buffers: 3 * block_size * 4 bytes. */
    bytes_lds = (long)3 * (long)spec->block_size * 4L;
    if(!ckc_archtarget_fits_lds(target, bytes_lds))
    {
        ln_set_reason(reason, reason_cap, "LDS budget exceeds cap");
        return false;
    }

    return true;
}

/* ------------------------------------------------------------------ *
 * tree_reduce combiner cookie: forward b.fadd as a ckc_combine_fn.
 * ------------------------------------------------------------------ */

static ckc_value_t* ln_fadd_combine(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c, void* user)
{
    (void)user;
    return ckc_b_fadd(b, a, c);
}

/* ------------------------------------------------------------------ *
 * pass-1 / pass-2 body contexts (Python closures over nonlocal state).
 * ------------------------------------------------------------------ */

typedef struct ln_pass1_ctx
{
    ckc_value_t* sum_p;
    ckc_value_t* sumsq_p;
} ln_pass1_ctx_t;

/* pass1_body(_n_off, x_scalars):
 *     sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
 *     sum_p   = b.fadd(sum_p,   tree_reduce(b, b.fadd, list(x_scalars)))
 *     sumsq_p = b.fadd(sumsq_p, tree_reduce(b, b.fadd, sq_scalars)) */
static void ln_pass1_body(
    ckc_ir_builder_t* b, ckc_value_t* n_off, ckc_value_t* const* x_scalars, int vec, void* user)
{
    ln_pass1_ctx_t* ctx = (ln_pass1_ctx_t*)user;
    /* sq_scalars (size vec); vec is one of {2,4,8}, small fixed bound. */
    ckc_value_t* sq[8];
    ckc_value_t* xs[8];
    ckc_value_t* part_sum;
    ckc_value_t* part_sumsq;
    int i;

    (void)n_off;
    if(vec > 8)
    {
        return; /* validate_io bounds vec to {2,4,8}; defensive */
    }

    for(i = 0; i < vec; ++i)
    {
        xs[i] = x_scalars[i];
        sq[i] = ckc_b_fmul(b, x_scalars[i], x_scalars[i]);
    }

    /* Python pass1_body fully evaluates the sum_p statement (x tree-reduce
     * then accumulate fadd) before the sumsq_p statement (sq tree-reduce then
     * accumulate fadd). The two accumulate fadds therefore interleave between
     * the two reduces; emit in that exact order. */
    part_sum = ckc_tree_reduce(b, ln_fadd_combine, NULL, xs, vec);
    ctx->sum_p = ckc_b_fadd(b, ctx->sum_p, part_sum);

    part_sumsq = ckc_tree_reduce(b, ln_fadd_combine, NULL, sq, vec);
    ctx->sumsq_p = ckc_b_fadd(b, ctx->sumsq_p, part_sumsq);
}

typedef struct ln_pass2_ctx
{
    bool two_pass;
    int vec;
    const ckc_tile_window_t* x_tile;
    const ckc_tensor_view_t* g_view;
    const ckc_tensor_view_t* b_view;
    ckc_value_t* mean;
    ckc_value_t* inv_std;
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
static void ln_pass2_body(ckc_ir_builder_t* b,
                          ckc_value_t* n_off,
                          int k,
                          ckc_value_t* const* x_scalars,
                          int num_x,
                          ckc_value_t** out,
                          int vec,
                          void* user)
{
    ln_pass2_ctx_t* ctx = (ln_pass2_ctx_t*)user;
    ckc_value_t* xs[8];
    ckc_value_t* gv[8];
    ckc_value_t* bv[8];
    ckc_value_t* idx1[1];
    int i;

    (void)k;
    if(vec > 8)
    {
        return;
    }

    if(ctx->two_pass)
    {
        /* x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC) */
        ckc_value_t* li[2];
        li[0] = ckc_b_const_i32(b, 0);
        li[1] = n_off;
        ckc_tile_window_load_vec_as_f32(b, ctx->x_tile, li, 2, vec, xs);
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
    ckc_tensor_view_load_vec_as_f32(b, ctx->g_view, idx1, 1, vec, gv);
    /* bv = b_view.load_vec_as_f32(b, [n_off], n=VEC) */
    ckc_tensor_view_load_vec_as_f32(b, ctx->b_view, idx1, 1, vec, bv);

    for(i = 0; i < vec; ++i)
    {
        ckc_value_t* dx = ckc_b_fsub(b, xs[i], ctx->mean);
        ckc_value_t* sg = ckc_b_fmul(b, ctx->inv_std, gv[i]);
        out[i] = ckc_b_fadd(b, ckc_b_fmul(b, dx, sg), bv[i]);
    }
}

/* ------------------------------------------------------------------ *
 * build_layernorm2d
 * ------------------------------------------------------------------ */

ckc_kernel_def_t* ckc_build_layernorm2d(ckc_ir_builder_t* b, const ckc_layernorm2d_spec_t* spec)
{
    const ckc_type_t* io_ty;
    int BS;
    int VEC;
    int N;
    int elems;
    bool two_pass;
    bool ok;

    ckc_value_t* X;
    ckc_value_t* Gamma;
    ckc_value_t* Beta;
    ckc_value_t* Y;
    ckc_value_t* Mean = NULL;
    ckc_value_t* InvStd = NULL;
    ckc_value_t* eps;

    ckc_value_t* tid;
    ckc_value_t* row;

    ckc_tensor_view_t x_view;
    ckc_tensor_view_t y_view;
    ckc_tensor_view_t g_view;
    ckc_tensor_view_t b_view;
    ckc_tile_window_t x_tile;
    ckc_tile_window_t y_tile;

    ckc_tensor_view_t lds_mean_v;
    ckc_tensor_view_t lds_m2_v;
    ckc_tensor_view_t lds_count_v;
    ckc_value_t* lds_mean;
    ckc_value_t* lds_m2;
    ckc_value_t* lds_count;

    ln_pass1_ctx_t p1;
    ln_pass2_ctx_t p2;
    ckc_row_chunk_sweep_result_t sweep_res;

    double count_p;
    ckc_value_t* inv_count_p;
    ckc_value_t* mean_p;
    ckc_value_t* m2_p;
    ckc_value_t* mean;
    ckc_value_t* var;
    ckc_value_t* inv_std;

    int shape2[2];
    int shape1[1];
    int ldsshape[1];
    ckc_value_t* origin[2];
    int lengths2[2];

    ckc_param_opts_t opts;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }

    /* ok, why = is_valid_spec(spec); if not ok: raise ValueError(...) */
    ok = ckc_layernorm2d_is_valid_spec(spec, "gfx950", NULL, 0);
    if(!ok)
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "%s", "invalid layernorm2d spec");
        return NULL;
    }

    io_ty = ckc_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    BS = spec->block_size;
    VEC = spec->vec;
    N = spec->n_per_block;
    elems = ckc_layernorm2d_elems_per_thread(spec);

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* --- params (ABI order matches CK Tile) --- */
    /* X = b.param("X", PtrType(io_ty,"global"), noalias, readonly, align=16) */
    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    X = ckc_b_param(b, "X", ckc_ptr_type(b, io_ty, "global"), &opts);

    Gamma = ckc_b_param(b, "Gamma", ckc_ptr_type(b, io_ty, "global"), &opts);
    Beta = ckc_b_param(b, "Beta", ckc_ptr_type(b, io_ty, "global"), &opts);

    /* Y = b.param("Y", PtrType(io_ty,"global"), noalias, writeonly, align=16) */
    memset(&opts, 0, sizeof opts);
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    Y = ckc_b_param(b, "Y", ckc_ptr_type(b, io_ty, "global"), &opts);

    if(spec->save_mean_invstd)
    {
        /* Mean / InvStd: noalias, writeonly (no align kwarg). */
        memset(&opts, 0, sizeof opts);
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        Mean = ckc_b_param(b, "Mean", ckc_ptr_type(b, io_ty, "global"), &opts);
        InvStd = ckc_b_param(b, "InvStd", ckc_ptr_type(b, io_ty, "global"), &opts);
    }

    /* M = b.param("M", I32); _ = b.param("N", I32); eps = b.param("eps", F32) */
    (void)ckc_b_param(b, "M", ckc_i32(), NULL);
    (void)ckc_b_param(b, "N", ckc_i32(), NULL);
    eps = ckc_b_param(b, "eps", ckc_f32(), NULL);

    /* tid = b.thread_id_x(); row = b.block_id_x() */
    tid = ckc_b_thread_id_x(b);
    row = ckc_b_block_id_x(b);

    /* --- CK Tile data abstractions --- */
    /* x_view = make_naive_tensor_view_packed(X, shape=(1,N), dtype=io_ty) */
    shape2[0] = 1;
    shape2[1] = N;
    ckc_make_naive_tensor_view_packed(&x_view, X, shape2, 2, io_ty);
    ckc_make_naive_tensor_view_packed(&y_view, Y, shape2, 2, io_ty);
    /* g_view = make_global_view(Gamma, shape=(N,), dtype=io_ty) */
    shape1[0] = N;
    ckc_make_global_view(&g_view, Gamma, shape1, 1, io_ty, NULL);
    ckc_make_global_view(&b_view, Beta, shape1, 1, io_ty, NULL);

    /* x_tile = make_tile_window(x_view, lengths=(1,N), origin=(row, 0)) */
    lengths2[0] = 1;
    lengths2[1] = N;
    origin[0] = row;
    origin[1] = ckc_b_const_i32(b, 0);
    ckc_make_tile_window(&x_tile, &x_view, lengths2, origin, 2);
    /* y_tile = make_tile_window(y_view, lengths=(1,N), origin=(row, 0)) */
    origin[0] = row;
    origin[1] = ckc_b_const_i32(b, 0);
    ckc_make_tile_window(&y_tile, &y_view, lengths2, origin, 2);

    /* --- LDS scratch (three f32 channels of BS words each) --- */
    ldsshape[0] = BS;
    ckc_make_lds_view(b, &lds_mean_v, ckc_f32(), ldsshape, 1, "lds_mean", NULL);
    ckc_make_lds_view(b, &lds_m2_v, ckc_f32(), ldsshape, 1, "lds_m2", NULL);
    ckc_make_lds_view(b, &lds_count_v, ckc_f32(), ldsshape, 1, "lds_count", NULL);
    lds_mean = lds_mean_v.base;
    lds_m2 = lds_m2_v.base;
    lds_count = lds_count_v.base;

    /* two_pass = row_norm_needs_two_pass(elems) */
    two_pass = ckc_row_norm_needs_two_pass(elems, CKC_REGISTER_TILE_MAX_ELEMS_PER_THREAD);

    /* sum_p = b.const_f32(0.0); sumsq_p = b.const_f32(0.0) */
    p1.sum_p = ckc_b_const_f32(b, 0.0);
    p1.sumsq_p = ckc_b_const_f32(b, 0.0);

    /* Pass 1: sweep_row_chunks(b, x_tile, tid, BS, VEC, elems,
     *                          body=pass1_body, cache=not two_pass) */
    sweep_res = ckc_sweep_row_chunks(b,
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
    /* count_p = float(elems); inv_count_p = b.const_f32(1.0 / count_p) */
    count_p = (double)elems;
    inv_count_p = ckc_b_const_f32(b, 1.0 / count_p);
    /* mean_p = b.fmul(sum_p, inv_count_p) */
    mean_p = ckc_b_fmul(b, p1.sum_p, inv_count_p);
    /* m2_p = b.fsub(sumsq_p, b.fmul(mean_p, sum_p)) */
    m2_p = ckc_b_fsub(b, p1.sumsq_p, ckc_b_fmul(b, mean_p, p1.sum_p));

    /* mean, var = welford_block_reduce_stable(b, mean_p, m2_p,
     *     b.const_f32(count_p), lds_mean, lds_m2, lds_count, tid,
     *     block_size=BS) */
    mean = NULL;
    var = NULL;
    ckc_welford_block_reduce_stable(b,
                                    mean_p,
                                    m2_p,
                                    ckc_b_const_f32(b, count_p),
                                    lds_mean,
                                    lds_m2,
                                    lds_count,
                                    tid,
                                    BS,
                                    &mean,
                                    &var);

    /* inv_std = b.rsqrt(b.fadd(var, eps)) */
    inv_std = ckc_b_rsqrt(b, ckc_b_fadd(b, var, eps));

    /* if save_mean_invstd:
     *     with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
     *         store_scalar_from_f32(b, Mean,   row, mean,    dtype)
     *         store_scalar_from_f32(b, InvStd, row, inv_std, dtype) */
    if(spec->save_mean_invstd)
    {
        ckc_if_t iff = ckc_b_scf_if(b, ckc_b_cmp_eq(b, tid, ckc_b_const_i32(b, 0)));
        ckc_b_region_enter(b, iff.then_region);
        ckc_b_store_scalar_from_f32(b, Mean, row, mean, spec->dtype);
        ckc_b_store_scalar_from_f32(b, InvStd, row, inv_std, spec->dtype);
        ckc_b_region_leave(b);
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

    ckc_pass2_row_chunks(b,
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
    if(!ckc_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

ckc_kernel_def_t* ckc_build_layernorm2d_new(ckc_ir_builder_t* b, const ckc_layernorm2d_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_layernorm2d_kernel_name(spec, name, sizeof name) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_layernorm2d(b, spec);
    });
}

/* ------------------------------------------------------------------ *
 * grid / signature
 * ------------------------------------------------------------------ */

ckc_status_t ckc_layernorm2d_grid(int m, const ckc_layernorm2d_spec_t* spec, int out[3])
{
    int totals[1];
    int tiles[1];
    (void)spec;
    if(out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    /* ceil_div_grid((m, 1)) -- one (total, tile) pair: (m, 1). */
    totals[0] = m;
    tiles[0] = 1;
    return ckc_ceil_div_grid(totals, tiles, 1, out);
}

/* layernorm2d_signature(spec):
 *   SignatureBuilder().ptr("X",dt).ptr("Gamma",dt).ptr("Beta",dt).ptr("Y",dt)
 *   [.ptr("Mean",dt).ptr("InvStd",dt)]
 *   .scalar("M","i32").scalar("N","i32").scalar("eps","f32").build() */
ckc_status_t ckc_layernorm2d_signature(ckc_signature_builder_t* sb,
                                       const ckc_layernorm2d_spec_t* spec,
                                       const ckc_sig_entry_t** out_items,
                                       size_t* out_count)
{
    if(sb == NULL || spec == NULL)
    {
        return CKC_ERR_VALUE;
    }
    ckc_signature_builder_ptr(sb, "X", spec->dtype, NULL);
    ckc_signature_builder_ptr(sb, "Gamma", spec->dtype, NULL);
    ckc_signature_builder_ptr(sb, "Beta", spec->dtype, NULL);
    ckc_signature_builder_ptr(sb, "Y", spec->dtype, NULL);
    if(spec->save_mean_invstd)
    {
        ckc_signature_builder_ptr(sb, "Mean", spec->dtype, NULL);
        ckc_signature_builder_ptr(sb, "InvStd", spec->dtype, NULL);
    }
    ckc_signature_builder_scalar(sb, "M", "i32");
    ckc_signature_builder_scalar(sb, "N", "i32");
    ckc_signature_builder_scalar(sb, "eps", "f32");
    return ckc_signature_builder_build(sb, out_items, out_count);
}

/* ------------------------------------------------------------------ *
 * lower-to-.ll convenience
 * ------------------------------------------------------------------ */

ckc_status_t ckc_layernorm2d_lower_to_llvm(const ckc_layernorm2d_spec_t* spec,
                                           const char* arch,
                                           ckc_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

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
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_layernorm2d_new(&b, spec);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            ln_set_reason(err, err_cap, m != NULL ? m : "build_layernorm2d failed");
        }
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
