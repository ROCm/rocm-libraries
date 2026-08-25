// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_rmsnorm2d.c -- C99 port of rocke/instances/common/rmsnorm2d.py.
 *
 * Byte-identical builder-call sequence vs the Python build_rmsnorm2d. The
 * Python lambda closures (pass1_body / pass2_body) become C function pointers
 * threaded with an explicit context struct, per the codebase convention used by
 * the sweep / persistent ports.
 */
#include "rocke/instance_rmsnorm2d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.sweep.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"

/* ------------------------------------------------------------------ peers *
 *
 * TensorView.load_vec_as_f32 (the g_view path in pass2_body) lives in the
 * tensor_view port. Declared here (like sweep.h declares the TileWindow peer)
 * so this TU compiles standalone; resolved at link time.
 *
 *   def load_vec_as_f32(self, b, indices, n) -> list[Value]
 *
 * Writes the n f32 SSA scalars to out[0..n) (caller-provided, length >= n). */
extern void rocke_tensor_view_load_vec_as_f32(rocke_ir_builder_t* b,
                                              const rocke_tensor_view_t* v,
                                              rocke_value_t* const* indices,
                                              int num_indices,
                                              int n,
                                              rocke_value_t** out);

/* ===================================================================== *
 *  RMSNorm2DSpec helpers (pure; no IR)
 * ===================================================================== */

rocke_rmsnorm2d_spec_t rocke_rmsnorm2d_spec_default(void)
{
    rocke_rmsnorm2d_spec_t s;
    s.n_per_block = 0;
    s.block_size = 256;
    s.vec = 4;
    s.dtype = "f16";
    s.save_inv_rms = false;
    s.wave_size = 64;
    s.name = "rocke_rmsnorm2d_fwd";
    return s;
}

int rocke_rmsnorm2d_elems_per_thread(const rocke_rmsnorm2d_spec_t* spec)
{
    /* n_per_block // block_size */
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return spec->n_per_block / spec->block_size;
}

rocke_status_t
    rocke_rmsnorm2d_kernel_name(const rocke_rmsnorm2d_spec_t* spec, char* out, size_t out_cap)
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
        return ROCKE_ERR_VALUE;
    }

    snprintf(part_n, sizeof(part_n), "N%d", spec->n_per_block);
    snprintf(part_b, sizeof(part_b), "b%d", spec->block_size);
    snprintf(part_v, sizeof(part_v), "v%d", spec->vec);

    parts[0] = spec->dtype;
    parts[1] = part_n;
    parts[2] = part_b;
    parts[3] = part_v;

    flag_names[0] = "sr";
    flag_on[0] = spec->save_inv_rms ? 1 : 0;

    return rocke_kernel_name_join(spec->name, parts, 4, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

bool rocke_rmsnorm2d_is_valid_spec(const rocke_rmsnorm2d_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap)
{
    const rocke_archtarget_t* target;
    int elems_per_thread;
    int two_pass;
    rocke_io_spec_rule_t rule;
    rocke_arena_t arena;
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
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unknown arch %s", arch);
        }
        return false;
    }

    elems_per_thread = rocke_rmsnorm2d_elems_per_thread(spec);

    /* cap = None if row_norm_needs_two_pass(...) else
     *       REGISTER_TILE_MAX_ELEMS_PER_THREAD */
    two_pass
        = rocke_row_norm_needs_two_pass(elems_per_thread, ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD)
              ? 1
              : 0;

    /* validate_io(IOSpecRule(dtype, block_size, vec, n_per_block,
     *                        max_elems_per_thread=cap)) */
    if(rocke_arena_init(&arena, 4096) != 0)
    {
        return false;
    }
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

    if(!rocke_validate_io(&arena, &rule, &why))
    {
        if(reason != NULL && reason_cap > 0 && why != NULL)
        {
            snprintf(reason, reason_cap, "%s", why);
        }
        rocke_arena_destroy(&arena);
        return false;
    }

    /* if block_size > target.max_threads_per_block: reject. */
    max_tpb = rocke_archtarget_max_threads_per_block(target);
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
    if(!rocke_archtarget_fits_lds(target, bytes_lds))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "LDS budget %ld over cap on %s", bytes_lds, arch);
        }
        ok = false;
        goto done;
    }

done:
    rocke_arena_destroy(&arena);
    return ok;
}

/* ===================================================================== *
 *  closure contexts + body callbacks
 * ===================================================================== */

/* pass1_body(_n_off, x_scalars):
 *     chunk_sq = [b.fmul(xi, xi) for xi in x_scalars]
 *     s2 = b.fadd(s2, tree_reduce(b, b.fadd, chunk_sq)) */
typedef struct rocke_rms_pass1_ctx
{
    rocke_value_t** s2; /* nonlocal s2 (mutated in place) */
} rocke_rms_pass1_ctx_t;

/* tree_reduce combiner: b.fadd (the rocke_combine_fn signature has a user cookie
 * we ignore). */
static rocke_value_t*
    rocke_rms_fadd_combine(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, void* user)
{
    (void)user;
    return rocke_b_fadd(b, a, c);
}

static void rocke_rms_pass1_body(rocke_ir_builder_t* b,
                                 rocke_value_t* n_off,
                                 rocke_value_t* const* x_scalars,
                                 int vec,
                                 void* user)
{
    rocke_rms_pass1_ctx_t* ctx = (rocke_rms_pass1_ctx_t*)user;
    rocke_value_t** chunk_sq;
    rocke_value_t* reduced;
    int i;

    (void)n_off; /* Python _n_off (unused) */

    /* chunk_sq = [b.fmul(xi, xi) for xi in x_scalars] */
    chunk_sq = (rocke_value_t**)rocke_arena_alloc(
        &b->arena, (size_t)(vec > 0 ? vec : 1) * sizeof(rocke_value_t*));
    for(i = 0; i < vec; ++i)
    {
        chunk_sq[i] = rocke_b_fmul(b, x_scalars[i], x_scalars[i]);
    }

    /* tree_reduce(b, b.fadd, chunk_sq) then s2 = b.fadd(s2, <reduced>) */
    reduced = rocke_tree_reduce(b, rocke_rms_fadd_combine, NULL, chunk_sq, vec);
    *ctx->s2 = rocke_b_fadd(b, *ctx->s2, reduced);
}

/* pass2_body(n_off, _k, x_scalars):
 *     if two_pass:
 *         x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
 *     gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
 *     return [b.fmul(x_scalars[i], b.fmul(inv_rms, gv[i])) for i in range(VEC)] */
typedef struct rocke_rms_pass2_ctx
{
    bool two_pass;
    const rocke_tile_window_t* x_tile;
    const rocke_tensor_view_t* g_view;
    rocke_value_t* inv_rms;
    int vec;
} rocke_rms_pass2_ctx_t;

static void rocke_rms_pass2_body(rocke_ir_builder_t* b,
                                 rocke_value_t* n_off,
                                 int k,
                                 rocke_value_t* const* x_scalars,
                                 int num_x,
                                 rocke_value_t** out,
                                 int vec,
                                 void* user)
{
    rocke_rms_pass2_ctx_t* ctx = (rocke_rms_pass2_ctx_t*)user;
    rocke_value_t** xs_local = NULL; /* two-pass freshly-loaded scalars */
    rocke_value_t* const* xs;
    rocke_value_t** gv;
    rocke_value_t* zero;
    int i;

    (void)k;
    (void)num_x;

    if(ctx->two_pass)
    {
        /* x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC) */
        rocke_value_t* local_indices[2];
        xs_local = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)(vec > 0 ? vec : 1) * sizeof(rocke_value_t*));
        zero = rocke_b_const_i32(b, 0);
        local_indices[0] = zero;
        local_indices[1] = n_off;
        rocke_tile_window_load_vec_as_f32(b, ctx->x_tile, local_indices, 2, vec, xs_local);
        xs = xs_local;
    }
    else
    {
        xs = x_scalars; /* cached f32 from pass 1 */
    }

    /* gv = g_view.load_vec_as_f32(b, [n_off], n=VEC) */
    gv = (rocke_value_t**)rocke_arena_alloc(&b->arena,
                                            (size_t)(vec > 0 ? vec : 1) * sizeof(rocke_value_t*));
    {
        rocke_value_t* g_indices[1];
        g_indices[0] = n_off;
        rocke_tensor_view_load_vec_as_f32(b, ctx->g_view, g_indices, 1, vec, gv);
    }

    /* return [b.fmul(x[i], b.fmul(inv_rms, gv[i])) for i in range(VEC)] */
    for(i = 0; i < vec; ++i)
    {
        out[i] = rocke_b_fmul(b, xs[i], rocke_b_fmul(b, ctx->inv_rms, gv[i]));
    }
}

/* ===================================================================== *
 *  build_rmsnorm2d(spec)
 * ===================================================================== */

rocke_kernel_def_t* rocke_build_rmsnorm2d(rocke_ir_builder_t* b,
                                          const rocke_rmsnorm2d_spec_t* spec,
                                          const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* io_ty;
        int BS, VEC, N;
        int elems_per_thread;
        int two_pass;

        rocke_value_t* X;
        rocke_value_t* Gamma;
        rocke_value_t* Y;
        rocke_value_t* InvRms = NULL;
        rocke_value_t* eps;

        rocke_value_t* tid;
        rocke_value_t* row;

        rocke_tensor_view_t x_view;
        rocke_tensor_view_t y_view;
        rocke_tensor_view_t g_view;
        rocke_tile_window_t x_tile;
        rocke_tile_window_t y_tile;
        rocke_value_t* lds;

        rocke_value_t* s2;
        rocke_rms_pass1_ctx_t p1ctx;
        rocke_row_chunk_sweep_result_t sweep_res;

        rocke_value_t* total_s2;
        rocke_value_t* rcp_n;
        rocke_value_t* mean_sq;
        rocke_value_t* inv_rms;

        rocke_rms_pass2_ctx_t p2ctx;

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
        if(!rocke_rmsnorm2d_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid rmsnorm2d spec: %s", reason);
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

        /* b.kernel.attrs["max_workgroup_size"] = BS */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

        /* ----- params (in Python order) ----- */
        {
            rocke_param_opts_t opts;
            const rocke_type_t* ptr_ty = rocke_ptr_type(b, io_ty, "global");

            /* X: noalias, readonly, align=16 */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.readonly = true;
            opts.readonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            X = rocke_b_param(b, "X", ptr_ty, &opts);

            /* Gamma: noalias, readonly, align=16 */
            Gamma = rocke_b_param(b, "Gamma", ptr_ty, &opts);

            /* Y: noalias, writeonly, align=16 */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.writeonly = true;
            opts.writeonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            Y = rocke_b_param(b, "Y", ptr_ty, &opts);

            /* InvRms: noalias, writeonly (no align kwarg) */
            if(spec->save_inv_rms)
            {
                memset(&opts, 0, sizeof(opts));
                opts.noalias = true;
                opts.noalias_set = true;
                opts.writeonly = true;
                opts.writeonly_set = true;
                InvRms = rocke_b_param(b, "InvRms", ptr_ty, &opts);
            }

            /* M : i32 (unused), N : i32 (unused), eps : f32 */
            (void)rocke_b_param(b, "M", rocke_i32(), NULL);
            (void)rocke_b_param(b, "N", rocke_i32(), NULL);
            eps = rocke_b_param(b, "eps", rocke_f32(), NULL);
        }

        tid = rocke_b_thread_id_x(b);
        row = rocke_b_block_id_x(b);

        /* x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
         *   (== make_global_view with packed strides). */
        {
            int shape2[2];
            shape2[0] = 1;
            shape2[1] = N;
            if(rocke_make_global_view(&x_view, X, shape2, 2, io_ty, NULL) != ROCKE_OK)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "rmsnorm2d: bad x_view");
                return NULL;
            }
            if(rocke_make_global_view(&y_view, Y, shape2, 2, io_ty, NULL) != ROCKE_OK)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "rmsnorm2d: bad y_view");
                return NULL;
            }
        }

        /* g_view = make_global_view(Gamma, shape=(N,), dtype=io_ty) */
        {
            int shape1[1];
            shape1[0] = N;
            if(rocke_make_global_view(&g_view, Gamma, shape1, 1, io_ty, NULL) != ROCKE_OK)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "rmsnorm2d: bad g_view");
                return NULL;
            }
        }

        /* x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, const_i32(0)))
         * y_tile = make_tile_window(y_view, lengths=(1, N), origin=(row, const_i32(0))) */
        {
            int lengths[2];
            rocke_value_t* origin[2];
            lengths[0] = 1;
            lengths[1] = N;
            origin[0] = row;
            origin[1] = rocke_b_const_i32(b, 0);
            if(rocke_make_tile_window(&x_tile, &x_view, lengths, origin, 2) != ROCKE_OK)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "rmsnorm2d: bad x_tile");
                return NULL;
            }
            origin[0] = row;
            origin[1] = rocke_b_const_i32(b, 0);
            if(rocke_make_tile_window(&y_tile, &y_view, lengths, origin, 2) != ROCKE_OK)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "rmsnorm2d: bad y_tile");
                return NULL;
            }
        }

        /* lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base
         *   (== b.smem_alloc(F32, [BS], name_hint="lds_red")). */
        {
            int lds_shape[1];
            lds_shape[0] = BS;
            lds = rocke_b_smem_alloc(b, rocke_f32(), lds_shape, 1, "lds_red");
        }

        /* two_pass = row_norm_needs_two_pass(spec.elems_per_thread) */
        elems_per_thread = rocke_rmsnorm2d_elems_per_thread(spec);
        two_pass = rocke_row_norm_needs_two_pass(elems_per_thread,
                                                 ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD)
                       ? 1
                       : 0;

        /* s2 = b.const_f32(0.0) */
        s2 = rocke_b_const_f32(b, 0.0);

        /* sweep_res = sweep_row_chunks(b, x_tile, tid, BS, VEC, elems_per_thread,
         *                              body=pass1_body, cache=not two_pass)
         *   (row defaults to None: the x_tile already carries the row origin). */
        p1ctx.s2 = &s2;
        sweep_res = rocke_sweep_row_chunks(b,
                                           &x_tile,
                                           tid,
                                           BS,
                                           VEC,
                                           elems_per_thread,
                                           /*row=*/NULL,
                                           rocke_rms_pass1_body,
                                           &p1ctx,
                                           /*cache=*/two_pass ? false : true);

        /* Cross-thread reduction. */
        if(spec->wave_size != 0 && (BS % spec->wave_size) == 0)
        {
            total_s2 = rocke_block_lds_reduce_with_wave_prologue(
                b, s2, lds, tid, BS, ROCKE_REDUCE_SUM, spec->wave_size);
        }
        else
        {
            total_s2 = rocke_block_lds_reduce(b, s2, lds, tid, BS, ROCKE_REDUCE_SUM);
        }

        /* rcp_n = b.rcp(b.const_f32(float(N)))
         * mean_sq = b.fmul(total_s2, rcp_n)
         * inv_rms = b.rsqrt(b.fadd(mean_sq, eps)) */
        rcp_n = rocke_b_rcp(b, rocke_b_const_f32(b, (double)N));
        mean_sq = rocke_b_fmul(b, total_s2, rcp_n);
        inv_rms = rocke_b_rsqrt(b, rocke_b_fadd(b, mean_sq, eps));

        /* if save_inv_rms:
         *     with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
         *         store_scalar_from_f32(b, InvRms, row, inv_rms, dtype=spec.dtype) */
        if(spec->save_inv_rms)
        {
            rocke_if_t gate = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
            rocke_b_region_enter(b, gate.then_region);
            rocke_b_store_scalar_from_f32(b, InvRms, row, inv_rms, spec->dtype);
            rocke_b_region_leave(b);
        }

        /* Pass 2: pass2_row_chunks(b, y_tile, tid, BS, VEC, elems_per_thread,
         *                          body=pass2_body, cached_f32=sweep_res.cached) */
        p2ctx.two_pass = two_pass ? true : false;
        p2ctx.x_tile = &x_tile;
        p2ctx.g_view = &g_view;
        p2ctx.inv_rms = inv_rms;
        p2ctx.vec = VEC;
        rocke_pass2_row_chunks(b,
                               &y_tile,
                               tid,
                               BS,
                               VEC,
                               elems_per_thread,
                               /*row=*/NULL,
                               rocke_rms_pass2_body,
                               &p2ctx,
                               sweep_res.cached,
                               sweep_res.num_cached);

        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_rmsnorm2d_new(rocke_ir_builder_t* b,
                                              const rocke_rmsnorm2d_spec_t* spec,
                                              const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_rmsnorm2d_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_rmsnorm2d(b, spec, arch);
    });
}

/* ===================================================================== *
 *  rmsnorm2d_grid(m, spec) -> (m, 1, 1)
 * ===================================================================== */

rocke_status_t rocke_rmsnorm2d_grid(int m, const rocke_rmsnorm2d_spec_t* spec, int out[3])
{
    int totals[2];
    int tiles[2];

    (void)spec; /* Python rmsnorm2d_grid ignores spec; ceil_div_grid((m, 1)). */

    if(out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    totals[0] = m;
    tiles[0] = 1;
    totals[1] = 1;
    tiles[1] = 1;
    return rocke_ceil_div_grid(totals, tiles, 2, out);
}

/* ===================================================================== *
 *  rmsnorm2d_signature(spec)
 * ===================================================================== */

rocke_status_t rocke_rmsnorm2d_signature(rocke_arena_t* arena,
                                         const rocke_rmsnorm2d_spec_t* spec,
                                         const rocke_sig_entry_t** out_items,
                                         size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t st;

    if(arena == NULL || spec == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }

    rocke_signature_builder_ptr(&sb, "X", spec->dtype, "global");
    rocke_signature_builder_ptr(&sb, "Gamma", spec->dtype, "global");
    rocke_signature_builder_ptr(&sb, "Y", spec->dtype, "global");
    if(spec->save_inv_rms)
    {
        rocke_signature_builder_ptr(&sb, "InvRms", spec->dtype, "global");
    }
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "eps", "f32");

    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  rocke_rmsnorm2d_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */

rocke_status_t rocke_rmsnorm2d_lower_to_llvm(const rocke_rmsnorm2d_spec_t* spec,
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
            const char* m = "lower_to_llvm: null spec/out";
            size_t n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_rmsnorm2d_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
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
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
