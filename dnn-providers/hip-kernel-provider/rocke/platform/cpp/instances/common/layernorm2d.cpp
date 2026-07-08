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
 *                    f"v{vec}", flags={"smv": save_mean_invstd}) */
rocke_status_t
    rocke_layernorm2d_kernel_name(const rocke_layernorm2d_spec_t* spec, char* out, size_t out_cap)
{
    char part_n[32];
    char part_b[32];
    char part_v[32];
    const char* parts[4];
    const char* flag_names[1];
    int flag_on[1];

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

    return rocke_kernel_name_join(spec->name, parts, 4, flag_names, flag_on, 1, out, out_cap, NULL);
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
    mean = NULL;
    var = NULL;
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
