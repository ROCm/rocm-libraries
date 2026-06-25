// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_rmsnorm2d.c -- C99 port of ck_dsl/instances/common/rmsnorm2d.py.
 *
 * Byte-identical builder-call sequence vs the Python build_rmsnorm2d. The
 * Python lambda closures (pass1_body / pass2_body) become C function pointers
 * threaded with an explicit context struct, per the codebase convention used by
 * the sweep / persistent ports.
 */
#include "ckc/instance_rmsnorm2d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */

#include "ckc/helper_ck_dsl.core.arch.h"
#include "ckc/helper_ck_dsl.helpers.io.h"
#include "ckc/helper_ck_dsl.helpers.reduction.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/helper_ck_dsl.helpers.sweep.h"
#include "ckc/helper_ck_dsl.helpers.tensor_view.h"
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* ------------------------------------------------------------------ peers *
 *
 * TensorView.load_vec_as_f32 (the g_view path in pass2_body) lives in the
 * tensor_view port. Declared here (like sweep.h declares the TileWindow peer)
 * so this TU compiles standalone; resolved at link time.
 *
 *   def load_vec_as_f32(self, b, indices, n) -> list[Value]
 *
 * Writes the n f32 SSA scalars to out[0..n) (caller-provided, length >= n). */
extern void ckc_tensor_view_load_vec_as_f32(ckc_ir_builder_t* b,
                                            const ckc_tensor_view_t* v,
                                            ckc_value_t* const* indices,
                                            int num_indices,
                                            int n,
                                            ckc_value_t** out);

/* ===================================================================== *
 *  RMSNorm2DSpec helpers (pure; no IR)
 * ===================================================================== */

ckc_rmsnorm2d_spec_t ckc_rmsnorm2d_spec_default(void)
{
    ckc_rmsnorm2d_spec_t s;
    s.n_per_block  = 0;
    s.block_size   = 256;
    s.vec          = 4;
    s.dtype        = "f16";
    s.save_inv_rms = false;
    s.wave_size    = 64;
    s.name         = "ck_dsl_rmsnorm2d_fwd";
    return s;
}

int ckc_rmsnorm2d_elems_per_thread(const ckc_rmsnorm2d_spec_t* spec)
{
    /* n_per_block // block_size */
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return spec->n_per_block / spec->block_size;
}

ckc_status_t ckc_rmsnorm2d_kernel_name(const ckc_rmsnorm2d_spec_t* spec, char* out, size_t out_cap)
{
    /* kernel_name_join(name, dtype, f"N{n_per_block}", f"b{block_size}",
     *                  f"v{vec}", flags={"sr": save_inv_rms}) */
    char part_n[32];
    char part_b[32];
    char part_v[32];
    const char* parts[4];
    const char* flag_names[1];
    int flag_on[1];

    if(spec == NULL)
    {
        return CKC_ERR_VALUE;
    }

    snprintf(part_n, sizeof(part_n), "N%d", spec->n_per_block);
    snprintf(part_b, sizeof(part_b), "b%d", spec->block_size);
    snprintf(part_v, sizeof(part_v), "v%d", spec->vec);

    parts[0] = spec->dtype;
    parts[1] = part_n;
    parts[2] = part_b;
    parts[3] = part_v;

    flag_names[0] = "sr";
    flag_on[0]    = spec->save_inv_rms ? 1 : 0;

    return ckc_kernel_name_join(spec->name, parts, 4, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

bool ckc_rmsnorm2d_is_valid_spec(const ckc_rmsnorm2d_spec_t* spec,
                                 const char* arch,
                                 char* reason,
                                 size_t reason_cap)
{
    const ckc_archtarget_t* target;
    int elems_per_thread;
    int two_pass;
    ckc_io_spec_rule_t rule;
    ckc_arena_t arena;
    const char* why = NULL;
    long bytes_lds;
    int max_tpb;
    bool ok = true;

    if(reason != NULL && reason_cap > 0)
    {
        reason[0] = '\0';
    }
    if(spec == NULL)
    {
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* target = ArchTarget.from_gfx(arch); KeyError -> (False, str(e)). */
    target = ckc_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unknown arch %s", arch);
        }
        return false;
    }

    elems_per_thread = ckc_rmsnorm2d_elems_per_thread(spec);

    /* cap = None if row_norm_needs_two_pass(...) else
     *       REGISTER_TILE_MAX_ELEMS_PER_THREAD */
    two_pass = ckc_row_norm_needs_two_pass(elems_per_thread, CKC_REGISTER_TILE_MAX_ELEMS_PER_THREAD)
                   ? 1
                   : 0;

    /* validate_io(IOSpecRule(dtype, block_size, vec, n_per_block,
     *                        max_elems_per_thread=cap)) */
    if(ckc_arena_init(&arena, 4096) != 0)
    {
        return false;
    }
    ckc_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec);
    rule.n_per_block_set = 1;
    rule.n_per_block     = spec->n_per_block;
    if(two_pass)
    {
        rule.max_elems_per_thread_set = 0; /* None */
    }
    else
    {
        rule.max_elems_per_thread_set = 1;
        rule.max_elems_per_thread     = CKC_REGISTER_TILE_MAX_ELEMS_PER_THREAD;
    }

    if(!ckc_validate_io(&arena, &rule, &why))
    {
        if(reason != NULL && reason_cap > 0 && why != NULL)
        {
            snprintf(reason, reason_cap, "%s", why);
        }
        ckc_arena_destroy(&arena);
        return false;
    }

    /* if block_size > target.max_threads_per_block: reject. */
    max_tpb = ckc_archtarget_max_threads_per_block(target);
    if(spec->block_size > max_tpb)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "block_size %d > max_threads_per_block %d on %s",
                     spec->block_size,
                     max_tpb,
                     arch);
        }
        ok = false;
        goto done;
    }

    /* One f32 LDS reduction buffer of block_size words. */
    bytes_lds = (long)spec->block_size * 4;
    if(!ckc_archtarget_fits_lds(target, bytes_lds))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "LDS budget %ld over cap on %s", bytes_lds, arch);
        }
        ok = false;
        goto done;
    }

done:
    ckc_arena_destroy(&arena);
    return ok;
}

/* ===================================================================== *
 *  closure contexts + body callbacks
 * ===================================================================== */

/* pass1_body(_n_off, x_scalars):
 *     chunk_sq = [b.fmul(xi, xi) for xi in x_scalars]
 *     s2 = b.fadd(s2, tree_reduce(b, b.fadd, chunk_sq)) */
typedef struct ckc_rms_pass1_ctx
{
    ckc_value_t** s2; /* nonlocal s2 (mutated in place) */
} ckc_rms_pass1_ctx_t;

/* tree_reduce combiner: b.fadd (the ckc_combine_fn signature has a user cookie
 * we ignore). */
static ckc_value_t*
ckc_rms_fadd_combine(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c, void* user)
{
    (void)user;
    return ckc_b_fadd(b, a, c);
}

static void ckc_rms_pass1_body(
    ckc_ir_builder_t* b, ckc_value_t* n_off, ckc_value_t* const* x_scalars, int vec, void* user)
{
    ckc_rms_pass1_ctx_t* ctx = (ckc_rms_pass1_ctx_t*)user;
    ckc_value_t** chunk_sq;
    ckc_value_t* reduced;
    int i;

    (void)n_off; /* Python _n_off (unused) */

    /* chunk_sq = [b.fmul(xi, xi) for xi in x_scalars] */
    chunk_sq = (ckc_value_t**)ckc_arena_alloc(&b->arena,
                                              (size_t)(vec > 0 ? vec : 1) * sizeof(ckc_value_t*));
    for(i = 0; i < vec; ++i)
    {
        chunk_sq[i] = ckc_b_fmul(b, x_scalars[i], x_scalars[i]);
    }

    /* tree_reduce(b, b.fadd, chunk_sq) then s2 = b.fadd(s2, <reduced>) */
    reduced  = ckc_tree_reduce(b, ckc_rms_fadd_combine, NULL, chunk_sq, vec);
    *ctx->s2 = ckc_b_fadd(b, *ctx->s2, reduced);
}

/* pass2_body(n_off, _k, x_scalars):
 *     if two_pass:
 *         x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
 *     gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
 *     return [b.fmul(x_scalars[i], b.fmul(inv_rms, gv[i])) for i in range(VEC)] */
typedef struct ckc_rms_pass2_ctx
{
    bool two_pass;
    const ckc_tile_window_t* x_tile;
    const ckc_tensor_view_t* g_view;
    ckc_value_t* inv_rms;
    int vec;
} ckc_rms_pass2_ctx_t;

static void ckc_rms_pass2_body(ckc_ir_builder_t* b,
                               ckc_value_t* n_off,
                               int k,
                               ckc_value_t* const* x_scalars,
                               int num_x,
                               ckc_value_t** out,
                               int vec,
                               void* user)
{
    ckc_rms_pass2_ctx_t* ctx = (ckc_rms_pass2_ctx_t*)user;
    ckc_value_t** xs_local   = NULL; /* two-pass freshly-loaded scalars */
    ckc_value_t* const* xs;
    ckc_value_t** gv;
    ckc_value_t* zero;
    int i;

    (void)k;
    (void)num_x;

    if(ctx->two_pass)
    {
        /* x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC) */
        ckc_value_t* local_indices[2];
        xs_local = (ckc_value_t**)ckc_arena_alloc(
            &b->arena, (size_t)(vec > 0 ? vec : 1) * sizeof(ckc_value_t*));
        zero             = ckc_b_const_i32(b, 0);
        local_indices[0] = zero;
        local_indices[1] = n_off;
        ckc_tile_window_load_vec_as_f32(b, ctx->x_tile, local_indices, 2, vec, xs_local);
        xs = xs_local;
    }
    else
    {
        xs = x_scalars; /* cached f32 from pass 1 */
    }

    /* gv = g_view.load_vec_as_f32(b, [n_off], n=VEC) */
    gv = (ckc_value_t**)ckc_arena_alloc(&b->arena,
                                        (size_t)(vec > 0 ? vec : 1) * sizeof(ckc_value_t*));
    {
        ckc_value_t* g_indices[1];
        g_indices[0] = n_off;
        ckc_tensor_view_load_vec_as_f32(b, ctx->g_view, g_indices, 1, vec, gv);
    }

    /* return [b.fmul(x[i], b.fmul(inv_rms, gv[i])) for i in range(VEC)] */
    for(i = 0; i < vec; ++i)
    {
        out[i] = ckc_b_fmul(b, xs[i], ckc_b_fmul(b, ctx->inv_rms, gv[i]));
    }
}

/* ===================================================================== *
 *  build_rmsnorm2d(spec)
 * ===================================================================== */

ckc_kernel_def_t*
ckc_build_rmsnorm2d(ckc_ir_builder_t* b, const ckc_rmsnorm2d_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        const ckc_type_t* io_ty;
        int BS, VEC, N;
        int elems_per_thread;
        int two_pass;

        ckc_value_t* X;
        ckc_value_t* Gamma;
        ckc_value_t* Y;
        ckc_value_t* InvRms = NULL;
        ckc_value_t* eps;

        ckc_value_t* tid;
        ckc_value_t* row;

        ckc_tensor_view_t x_view;
        ckc_tensor_view_t y_view;
        ckc_tensor_view_t g_view;
        ckc_tile_window_t x_tile;
        ckc_tile_window_t y_tile;
        ckc_value_t* lds;

        ckc_value_t* s2;
        ckc_rms_pass1_ctx_t p1ctx;
        ckc_row_chunk_sweep_result_t sweep_res;

        ckc_value_t* total_s2;
        ckc_value_t* rcp_n;
        ckc_value_t* mean_sq;
        ckc_value_t* inv_rms;

        ckc_rms_pass2_ctx_t p2ctx;

        char reason[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx950";
        }

        /* ok, why = is_valid_spec(spec); raise ValueError on reject. */
        if(!ckc_rmsnorm2d_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "invalid rmsnorm2d spec: %s", reason);
            return NULL;
        }

        io_ty = ckc_b_io_ir_type(b, spec->dtype);
        if(io_ty == NULL)
        {
            return NULL;
        }
        BS  = spec->block_size;
        VEC = spec->vec;
        N   = spec->n_per_block;

        /* b.kernel.attrs["max_workgroup_size"] = BS */
        ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

        /* ----- params (in Python order) ----- */
        {
            ckc_param_opts_t opts;
            const ckc_type_t* ptr_ty = ckc_ptr_type(b, io_ty, "global");

            /* X: noalias, readonly, align=16 */
            memset(&opts, 0, sizeof(opts));
            opts.noalias      = true;
            opts.noalias_set  = true;
            opts.readonly     = true;
            opts.readonly_set = true;
            opts.align        = 16;
            opts.align_set    = true;
            X                 = ckc_b_param(b, "X", ptr_ty, &opts);

            /* Gamma: noalias, readonly, align=16 */
            Gamma = ckc_b_param(b, "Gamma", ptr_ty, &opts);

            /* Y: noalias, writeonly, align=16 */
            memset(&opts, 0, sizeof(opts));
            opts.noalias       = true;
            opts.noalias_set   = true;
            opts.writeonly     = true;
            opts.writeonly_set = true;
            opts.align         = 16;
            opts.align_set     = true;
            Y                  = ckc_b_param(b, "Y", ptr_ty, &opts);

            /* InvRms: noalias, writeonly (no align kwarg) */
            if(spec->save_inv_rms)
            {
                memset(&opts, 0, sizeof(opts));
                opts.noalias       = true;
                opts.noalias_set   = true;
                opts.writeonly     = true;
                opts.writeonly_set = true;
                InvRms             = ckc_b_param(b, "InvRms", ptr_ty, &opts);
            }

            /* M : i32 (unused), N : i32 (unused), eps : f32 */
            (void)ckc_b_param(b, "M", ckc_i32(), NULL);
            (void)ckc_b_param(b, "N", ckc_i32(), NULL);
            eps = ckc_b_param(b, "eps", ckc_f32(), NULL);
        }

        tid = ckc_b_thread_id_x(b);
        row = ckc_b_block_id_x(b);

        /* x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
         *   (== make_global_view with packed strides). */
        {
            int shape2[2];
            shape2[0] = 1;
            shape2[1] = N;
            if(ckc_make_global_view(&x_view, X, shape2, 2, io_ty, NULL) != CKC_OK)
            {
                ckc_i_set_err(b, CKC_ERR_VALUE, "rmsnorm2d: bad x_view");
                return NULL;
            }
            if(ckc_make_global_view(&y_view, Y, shape2, 2, io_ty, NULL) != CKC_OK)
            {
                ckc_i_set_err(b, CKC_ERR_VALUE, "rmsnorm2d: bad y_view");
                return NULL;
            }
        }

        /* g_view = make_global_view(Gamma, shape=(N,), dtype=io_ty) */
        {
            int shape1[1];
            shape1[0] = N;
            if(ckc_make_global_view(&g_view, Gamma, shape1, 1, io_ty, NULL) != CKC_OK)
            {
                ckc_i_set_err(b, CKC_ERR_VALUE, "rmsnorm2d: bad g_view");
                return NULL;
            }
        }

        /* x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, const_i32(0)))
         * y_tile = make_tile_window(y_view, lengths=(1, N), origin=(row, const_i32(0))) */
        {
            int lengths[2];
            ckc_value_t* origin[2];
            lengths[0] = 1;
            lengths[1] = N;
            origin[0]  = row;
            origin[1]  = ckc_b_const_i32(b, 0);
            if(ckc_make_tile_window(&x_tile, &x_view, lengths, origin, 2) != CKC_OK)
            {
                ckc_i_set_err(b, CKC_ERR_VALUE, "rmsnorm2d: bad x_tile");
                return NULL;
            }
            origin[0] = row;
            origin[1] = ckc_b_const_i32(b, 0);
            if(ckc_make_tile_window(&y_tile, &y_view, lengths, origin, 2) != CKC_OK)
            {
                ckc_i_set_err(b, CKC_ERR_VALUE, "rmsnorm2d: bad y_tile");
                return NULL;
            }
        }

        /* lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base
         *   (== b.smem_alloc(F32, [BS], name_hint="lds_red")). */
        {
            int lds_shape[1];
            lds_shape[0] = BS;
            lds          = ckc_b_smem_alloc(b, ckc_f32(), lds_shape, 1, "lds_red");
        }

        /* two_pass = row_norm_needs_two_pass(spec.elems_per_thread) */
        elems_per_thread = ckc_rmsnorm2d_elems_per_thread(spec);
        two_pass =
            ckc_row_norm_needs_two_pass(elems_per_thread, CKC_REGISTER_TILE_MAX_ELEMS_PER_THREAD)
                ? 1
                : 0;

        /* s2 = b.const_f32(0.0) */
        s2 = ckc_b_const_f32(b, 0.0);

        /* sweep_res = sweep_row_chunks(b, x_tile, tid, BS, VEC, elems_per_thread,
         *                              body=pass1_body, cache=not two_pass)
         *   (row defaults to None: the x_tile already carries the row origin). */
        p1ctx.s2  = &s2;
        sweep_res = ckc_sweep_row_chunks(b,
                                         &x_tile,
                                         tid,
                                         BS,
                                         VEC,
                                         elems_per_thread,
                                         /*row=*/NULL,
                                         ckc_rms_pass1_body,
                                         &p1ctx,
                                         /*cache=*/two_pass ? false : true);

        /* Cross-thread reduction. */
        if(spec->wave_size != 0 && (BS % spec->wave_size) == 0)
        {
            total_s2 = ckc_block_lds_reduce_with_wave_prologue(
                b, s2, lds, tid, BS, CKC_REDUCE_SUM, spec->wave_size);
        }
        else
        {
            total_s2 = ckc_block_lds_reduce(b, s2, lds, tid, BS, CKC_REDUCE_SUM);
        }

        /* rcp_n = b.rcp(b.const_f32(float(N)))
         * mean_sq = b.fmul(total_s2, rcp_n)
         * inv_rms = b.rsqrt(b.fadd(mean_sq, eps)) */
        rcp_n   = ckc_b_rcp(b, ckc_b_const_f32(b, (double)N));
        mean_sq = ckc_b_fmul(b, total_s2, rcp_n);
        inv_rms = ckc_b_rsqrt(b, ckc_b_fadd(b, mean_sq, eps));

        /* if save_inv_rms:
         *     with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
         *         store_scalar_from_f32(b, InvRms, row, inv_rms, dtype=spec.dtype) */
        if(spec->save_inv_rms)
        {
            ckc_if_t gate = ckc_b_scf_if(b, ckc_b_cmp_eq(b, tid, ckc_b_const_i32(b, 0)));
            ckc_b_region_enter(b, gate.then_region);
            ckc_b_store_scalar_from_f32(b, InvRms, row, inv_rms, spec->dtype);
            ckc_b_region_leave(b);
        }

        /* Pass 2: pass2_row_chunks(b, y_tile, tid, BS, VEC, elems_per_thread,
         *                          body=pass2_body, cached_f32=sweep_res.cached) */
        p2ctx.two_pass = two_pass ? true : false;
        p2ctx.x_tile   = &x_tile;
        p2ctx.g_view   = &g_view;
        p2ctx.inv_rms  = inv_rms;
        p2ctx.vec      = VEC;
        ckc_pass2_row_chunks(b,
                             &y_tile,
                             tid,
                             BS,
                             VEC,
                             elems_per_thread,
                             /*row=*/NULL,
                             ckc_rms_pass2_body,
                             &p2ctx,
                             sweep_res.cached,
                             sweep_res.num_cached);

        return b->kernel;
    });
}

ckc_kernel_def_t*
ckc_build_rmsnorm2d_new(ckc_ir_builder_t* b, const ckc_rmsnorm2d_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_rmsnorm2d_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_rmsnorm2d(b, spec, arch);
    });
}

/* ===================================================================== *
 *  rmsnorm2d_grid(m, spec) -> (m, 1, 1)
 * ===================================================================== */

ckc_status_t ckc_rmsnorm2d_grid(int m, const ckc_rmsnorm2d_spec_t* spec, int out[3])
{
    int totals[2];
    int tiles[2];

    (void)spec; /* Python rmsnorm2d_grid ignores spec; ceil_div_grid((m, 1)). */

    if(out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    totals[0] = m;
    tiles[0]  = 1;
    totals[1] = 1;
    tiles[1]  = 1;
    return ckc_ceil_div_grid(totals, tiles, 2, out);
}

/* ===================================================================== *
 *  rmsnorm2d_signature(spec)
 * ===================================================================== */

ckc_status_t ckc_rmsnorm2d_signature(ckc_arena_t* arena,
                                     const ckc_rmsnorm2d_spec_t* spec,
                                     const ckc_sig_entry_t** out_items,
                                     size_t* out_count)
{
    ckc_signature_builder_t sb;
    ckc_status_t st;

    if(arena == NULL || spec == NULL || out_items == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }

    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
    {
        return st;
    }

    ckc_signature_builder_ptr(&sb, "X", spec->dtype, "global");
    ckc_signature_builder_ptr(&sb, "Gamma", spec->dtype, "global");
    ckc_signature_builder_ptr(&sb, "Y", spec->dtype, "global");
    if(spec->save_inv_rms)
    {
        ckc_signature_builder_ptr(&sb, "InvRms", spec->dtype, "global");
    }
    ckc_signature_builder_scalar(&sb, "M", "i32");
    ckc_signature_builder_scalar(&sb, "N", "i32");
    ckc_signature_builder_scalar(&sb, "eps", "f32");

    return ckc_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  ckc_rmsnorm2d_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */

ckc_status_t ckc_rmsnorm2d_lower_to_llvm(const ckc_rmsnorm2d_spec_t* spec,
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
            const char* m = "lower_to_llvm: null spec/out";
            size_t n      = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_rmsnorm2d_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_rmsnorm2d failed";
            }
            n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
