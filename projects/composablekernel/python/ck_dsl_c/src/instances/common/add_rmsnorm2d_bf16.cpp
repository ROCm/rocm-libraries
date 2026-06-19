// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_add_rmsnorm2d_bf16.c -- C99 port of
 * ck_dsl/instances/common/add_rmsnorm2d_bf16.py.
 *
 * Byte-faithful reproduction of the Python builder-call sequence: two passes
 * over the per-row tile (pass 1 streams A & B, computes x = a + b + per-thread
 * sum-of-squares and caches x; pass 2 normalizes y = x * inv_rms * gamma), with
 * an arch-selected cross-thread reduction in between.
 */
#include "ckc/instance_add_rmsnorm2d_bf16.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/helper_ck_dsl.core.arch.h"
#include "ckc/helper_ck_dsl.helpers.io.h"
#include "ckc/helper_ck_dsl.helpers.reduction.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/helper_ck_dsl.helpers.tensor_view.h"
#include "ckc/ir_internal.h"      /* ckc_i_set_err */
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */

ckc_add_rmsnorm2d_bf16_spec_t ckc_add_rmsnorm2d_bf16_spec_default(void)
{
    ckc_add_rmsnorm2d_bf16_spec_t s;
    s.n_per_block   = 0; /* mandatory; no Python default */
    s.block_size    = 256;
    s.vec           = 4;
    s.dtype         = "bf16";
    s.save_residual = true;
    s.wave_size     = 64;
    s.name          = "ck_dsl_add_rmsnorm2d_bf16";
    return s;
}

int ckc_add_rmsnorm2d_bf16_elems_per_thread(const ckc_add_rmsnorm2d_bf16_spec_t* spec)
{
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return spec->n_per_block / spec->block_size;
}

ckc_status_t ckc_add_rmsnorm2d_bf16_kernel_name(const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                                char* out,
                                                size_t out_cap)
{
    /* kernel_name_join(name, dtype, f"N{n}", f"b{bs}", f"v{vec}",
     *                  flags={"sr": save_residual}) */
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
    parts[0]      = spec->dtype;
    parts[1]      = part_n;
    parts[2]      = part_b;
    parts[3]      = part_v;
    flag_names[0] = "sr";
    flag_on[0]    = spec->save_residual ? 1 : 0;

    return ckc_kernel_name_join(spec->name, parts, 4, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */

bool ckc_is_valid_spec_add_rmsnorm2d_bf16(const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap)
{
    const ckc_archtarget_t* target;
    ckc_io_spec_rule_t rule;
    const char* why = NULL;
    ckc_arena_t arena;
    int ok_io;
    long bytes_lds;

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

    /* try: target = ArchTarget.from_gfx(arch) except KeyError: return False */
    target = ckc_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "%s", arch);
        }
        return false;
    }

    /* validate_io(IOSpecRule(dtype, block_size, vec, n_per_block,
     *                        max_elems_per_thread=64)) */
    if(ckc_arena_init(&arena, 4096) != CKC_OK)
    {
        return false;
    }
    ckc_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec);
    rule.n_per_block_set          = 1;
    rule.n_per_block              = spec->n_per_block;
    rule.max_elems_per_thread_set = 1;
    rule.max_elems_per_thread     = 64;
    ok_io                         = ckc_validate_io(&arena, &rule, &why);
    if(!ok_io)
    {
        if(reason != NULL && reason_cap > 0 && why != NULL)
        {
            snprintf(reason, reason_cap, "%s", why);
        }
        ckc_arena_destroy(&arena);
        return false;
    }

    /* if spec.block_size > target.max_threads_per_block: reject */
    if(spec->block_size > ckc_archtarget_max_threads_per_block(target))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "block_size %d > max_threads_per_block %d on %s",
                     spec->block_size,
                     ckc_archtarget_max_threads_per_block(target),
                     arch);
        }
        ckc_arena_destroy(&arena);
        return false;
    }

    /* One f32 LDS reduction buffer of block_size words. */
    bytes_lds = (long)spec->block_size * 4;
    if(!ckc_archtarget_fits_lds(target, bytes_lds))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "LDS budget %ld > %d cap on %s",
                     bytes_lds,
                     target->lds_capacity_bytes,
                     arch);
        }
        ckc_arena_destroy(&arena);
        return false;
    }

    ckc_arena_destroy(&arena);
    return true;
}

/* ===================================================================== *
 *  f32-promoting load / demoting store (TensorView.load_vec_as_f32 /
 *  TileWindow.store_vec_from_f32 analogues, inlined here since the C
 *  tensor_view port does not expose the compute-promoting variants).
 * ===================================================================== */

/* TileWindow.load_vec_as_f32(b, row, col, n=VEC) -> n f32 scalars into out.
 * Mirrors Python: load_vec then per-lane cast_to_f32(vec_extract). The
 * row/col local indices match the window rank (2). n is in {2,4,8} here. */
static void rms_tile_load_vec_as_f32(ckc_ir_builder_t* b,
                                     const ckc_tile_window_t* w,
                                     ckc_value_t* row,
                                     ckc_value_t* col,
                                     int n,
                                     ckc_value_t** out)
{
    ckc_value_t* idx[2];
    ckc_value_t* v;
    int i;

    idx[0] = row;
    idx[1] = col;
    v      = ckc_tile_window_load_vec(b, w, idx, 2, n);
    for(i = 0; i < n; ++i)
    {
        out[i] = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, v, i));
    }
}

/* TileWindow.store_vec_from_f32(b, row, col, values) -- f32 demote + pack +
 * vector store. Mirrors Python for len(values) in {2,4,8}. */
static void rms_tile_store_vec_from_f32(ckc_ir_builder_t* b,
                                        const ckc_tile_window_t* w,
                                        ckc_value_t* row,
                                        ckc_value_t* col,
                                        ckc_value_t* const* values,
                                        int n)
{
    ckc_value_t* idx[2];
    ckc_value_t* casts[8];
    ckc_value_t* packed;
    const ckc_type_t* dt = ckc_tile_window_dtype(w);
    int i;

    idx[0] = row;
    idx[1] = col;
    for(i = 0; i < n; ++i)
    {
        casts[i] = ckc_b_cast_f32_to(b, values[i], dt);
    }
    packed = ckc_b_vec_pack(b, casts, n, dt);
    ckc_tile_window_store_vec(b, w, idx, 2, packed, n);
}

/* TensorView.load_vec_as_f32(b, [n_off], n=VEC) -> n f32 scalars (Gamma view,
 * rank 1). */
static void rms_view_load_vec_as_f32(
    ckc_ir_builder_t* b, const ckc_tensor_view_t* v, ckc_value_t* n_off, int n, ckc_value_t** out)
{
    ckc_value_t* idx[1];
    ckc_value_t* vec;
    int i;

    idx[0] = n_off;
    vec    = ckc_tensor_view_load_vec(b, v, idx, 1, n);
    for(i = 0; i < n; ++i)
    {
        out[i] = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, vec, i));
    }
}

/* tree_reduce combine callback: b.fadd. */
static ckc_value_t*
rms_combine_fadd(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c, void* user)
{
    (void)user;
    return ckc_b_fadd(b, a, c);
}

/* ===================================================================== *
 *  build_add_rmsnorm2d_bf16
 * ===================================================================== */

ckc_kernel_def_t* ckc_build_add_rmsnorm2d_bf16(ckc_ir_builder_t* b,
                                               const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                               const char* arch)
{
    const ckc_type_t* io_ty;
    int BS, VEC, N, chunks_per_thread;
    const ckc_archtarget_t* hw;
    int hw_wave;

    ckc_value_t *A, *Bp, *Gamma, *X = NULL, *Y;
    ckc_value_t *tid, *row, *c_vec;
    ckc_tensor_view_t a_view, b_view, g_view, y_view, x_view;
    ckc_tile_window_t a_tile, bt_tile, y_tile, x_tile;
    ckc_value_t* lds;
    ckc_value_t* s_sq;
    ckc_value_t* total_sq;
    ckc_value_t *rcp_n, *mean_sq, *inv_rms, *eps;
    /* cached_x holds elems_per_thread f32 scalars (<= 64). */
    ckc_value_t* cached_x[CKC_REGISTER_TILE_MAX_ELEMS_PER_THREAD];
    int cached_count = 0;
    int k, i;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError */
    {
        char why[256];
        if(!ckc_is_valid_spec_add_rmsnorm2d_bf16(spec, arch, why, sizeof(why)))
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "invalid add_rmsnorm2d_bf16 spec: %s", why);
            return NULL;
        }
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

    /* ---- params (Values) ---- */
    {
        ckc_param_opts_t opts;
        const ckc_type_t* ptr_io = ckc_ptr_type(b, io_ty, "global");

        memset(&opts, 0, sizeof(opts));
        opts.noalias      = true;
        opts.noalias_set  = true;
        opts.readonly     = true;
        opts.readonly_set = true;
        opts.align        = 16;
        opts.align_set    = true;
        A                 = ckc_b_param(b, "A", ptr_io, &opts);
        Bp                = ckc_b_param(b, "B", ptr_io, &opts);
        Gamma             = ckc_b_param(b, "Gamma", ptr_io, &opts);

        if(spec->save_residual)
        {
            memset(&opts, 0, sizeof(opts));
            opts.noalias       = true;
            opts.noalias_set   = true;
            opts.writeonly     = true;
            opts.writeonly_set = true;
            opts.align         = 16;
            opts.align_set     = true;
            X                  = ckc_b_param(b, "X", ptr_io, &opts);
        }

        memset(&opts, 0, sizeof(opts));
        opts.noalias       = true;
        opts.noalias_set   = true;
        opts.writeonly     = true;
        opts.writeonly_set = true;
        opts.align         = 16;
        opts.align_set     = true;
        Y                  = ckc_b_param(b, "Y", ptr_io, &opts);

        (void)ckc_b_param(b, "M", ckc_i32(), NULL); /* ABI symmetry */
        (void)ckc_b_param(b, "N", ckc_i32(), NULL); /* validated by caller */
        eps = ckc_b_param(b, "eps", ckc_f32(), NULL);
    }

    tid = ckc_b_thread_id_x(b);
    row = ckc_b_block_id_x(b);

    /* Views + tile windows. shape=(1, N) row-major for A/B/X/Y; Gamma=(N,).
     *
     * Each make_tile_window in Python is called with a FRESH ``b.const_i32(0)``
     * as its origin's column coordinate -- the builder does not cache constants,
     * so every one of those zeros consumes its own SSA id. Reusing a single
     * shared ``zero`` here (as before) under-counted those ids by 3 (sr=true) /
     * 2 (sr=false), which shifted the lds_red SSA suffix and the entire body
     * numbering. Allocate one fresh origin-zero per window, in Python's order
     * (a_tile, bt_tile, y_tile, then x_tile), to pin the id sequence. */
    {
        int shape2[2];
        int shapeg[1];
        int lengths2[2];
        ckc_value_t* origin_a[2];
        ckc_value_t* origin_b[2];
        ckc_value_t* origin_y[2];
        ckc_value_t* origin_x[2];

        shape2[0]   = 1;
        shape2[1]   = N;
        shapeg[0]   = N;
        lengths2[0] = 1;
        lengths2[1] = N;

        ckc_make_global_view(&a_view, A, shape2, 2, io_ty, NULL);
        ckc_make_global_view(&b_view, Bp, shape2, 2, io_ty, NULL);
        ckc_make_global_view(&g_view, Gamma, shapeg, 1, io_ty, NULL);
        ckc_make_global_view(&y_view, Y, shape2, 2, io_ty, NULL);

        origin_a[0] = row;
        origin_a[1] = ckc_b_const_i32(b, 0);
        ckc_make_tile_window(&a_tile, &a_view, lengths2, origin_a, 2);

        origin_b[0] = row;
        origin_b[1] = ckc_b_const_i32(b, 0);
        ckc_make_tile_window(&bt_tile, &b_view, lengths2, origin_b, 2);

        origin_y[0] = row;
        origin_y[1] = ckc_b_const_i32(b, 0);
        ckc_make_tile_window(&y_tile, &y_view, lengths2, origin_y, 2);

        if(spec->save_residual)
        {
            ckc_make_global_view(&x_view, X, shape2, 2, io_ty, NULL);
            origin_x[0] = row;
            origin_x[1] = ckc_b_const_i32(b, 0);
            ckc_make_tile_window(&x_tile, &x_view, lengths2, origin_x, 2);
        }
    }

    /* lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base */
    {
        int lds_shape[1];
        lds_shape[0] = BS;
        lds          = ckc_b_smem_alloc_f32(b, lds_shape, 1, "lds_red");
    }

    /* ---- Pass 1: stream A & B, x = a + b, per-thread sum-of-squares ---- */
    s_sq              = ckc_b_const_f32(b, 0.0);
    c_vec             = ckc_b_const_i32(b, VEC);
    chunks_per_thread = ckc_add_rmsnorm2d_bf16_elems_per_thread(spec) / VEC;

    for(k = 0; k < chunks_per_thread; ++k)
    {
        ckc_value_t* n_off;
        ckc_value_t* mul_k;
        ckc_value_t* mul_tid;
        ckc_value_t* a_scalars[8];
        ckc_value_t* b_scalars[8];
        ckc_value_t* chunk_x[8];
        ckc_value_t* chunk_sq[8];
        ckc_value_t* part;

        /* n_off = (k*BS)*VEC + tid*VEC
         * Python evaluates b.add's args left-to-right: the (k*BS)*VEC mul is
         * emitted BEFORE the tid*VEC mul. C leaves argument evaluation order
         * unspecified (this compiler evaluates right-to-left), so emit the two
         * muls into temporaries first to pin the SSA-id order to Python's. */
        mul_k   = ckc_b_mul(b, ckc_b_const_i32(b, (int64_t)k * BS), c_vec);
        mul_tid = ckc_b_mul(b, tid, c_vec);
        n_off   = ckc_b_add(b, mul_k, mul_tid);

        /* Python passes a FRESH b.const_i32(0) as the row local-index on each
         * load/store call; mirror that here so the constant-id sequence lines up
         * with the reference builder. */
        rms_tile_load_vec_as_f32(b, &a_tile, ckc_b_const_i32(b, 0), n_off, VEC, a_scalars);
        rms_tile_load_vec_as_f32(b, &bt_tile, ckc_b_const_i32(b, 0), n_off, VEC, b_scalars);

        for(i = 0; i < VEC; ++i)
        {
            ckc_value_t* x_i = ckc_b_fadd(b, a_scalars[i], b_scalars[i]);
            chunk_x[i]       = x_i;
            chunk_sq[i]      = ckc_b_fmul(b, x_i, x_i);
        }
        /* s_sq = s_sq + tree_reduce(b, b.fadd, chunk_sq) */
        part = ckc_tree_reduce(b, rms_combine_fadd, NULL, chunk_sq, VEC);
        s_sq = ckc_b_fadd(b, s_sq, part);

        /* cached_x.extend(chunk_x) */
        for(i = 0; i < VEC; ++i)
        {
            if(cached_count < CKC_REGISTER_TILE_MAX_ELEMS_PER_THREAD)
            {
                cached_x[cached_count++] = chunk_x[i];
            }
        }

        if(spec->save_residual)
        {
            rms_tile_store_vec_from_f32(b, &x_tile, ckc_b_const_i32(b, 0), n_off, chunk_x, VEC);
        }
    }

    /* ---- Cross-thread reduction ---- */
    hw      = ckc_archtarget_from_gfx(arch);
    hw_wave = (hw != NULL) ? hw->wave_size : 0;
    if(hw_wave == spec->wave_size && (spec->block_size % spec->wave_size) == 0)
    {
        total_sq = ckc_block_lds_reduce_with_wave_prologue(
            b, s_sq, lds, tid, spec->block_size, CKC_REDUCE_SUM, spec->wave_size);
    }
    else
    {
        total_sq = ckc_block_lds_reduce(b, s_sq, lds, tid, BS, CKC_REDUCE_SUM);
    }

    rcp_n   = ckc_b_rcp(b, ckc_b_const_f32(b, (double)N));
    mean_sq = ckc_b_fmul(b, total_sq, rcp_n);
    inv_rms = ckc_b_rsqrt(b, ckc_b_fadd(b, mean_sq, eps));

    /* ---- Pass 2: y = x * (inv_rms * gamma) ---- */
    for(k = 0; k < chunks_per_thread; ++k)
    {
        ckc_value_t* n_off;
        ckc_value_t* mul_k;
        ckc_value_t* mul_tid;
        ckc_value_t* gv[8];
        ckc_value_t* y_vec[8];

        /* Same left-to-right arg-eval pinning as pass 1: (k*BS)*VEC mul first,
         * then tid*VEC mul, then the add. */
        mul_k   = ckc_b_mul(b, ckc_b_const_i32(b, (int64_t)k * BS), c_vec);
        mul_tid = ckc_b_mul(b, tid, c_vec);
        n_off   = ckc_b_add(b, mul_k, mul_tid);

        rms_view_load_vec_as_f32(b, &g_view, n_off, VEC, gv);
        for(i = 0; i < VEC; ++i)
        {
            y_vec[i] = ckc_b_fmul(b, cached_x[k * VEC + i], ckc_b_fmul(b, inv_rms, gv[i]));
        }
        rms_tile_store_vec_from_f32(b, &y_tile, ckc_b_const_i32(b, 0), n_off, y_vec, VEC);
    }

    return b->kernel;
}

ckc_kernel_def_t* ckc_build_add_rmsnorm2d_bf16_new(ckc_ir_builder_t* b,
                                                   const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                                   const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_add_rmsnorm2d_bf16_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_add_rmsnorm2d_bf16(b, spec, arch);
    });
}

/* ===================================================================== *
 *  grid
 * ===================================================================== */

ckc_status_t
ckc_add_rmsnorm2d_bf16_grid(int m, const ckc_add_rmsnorm2d_bf16_spec_t* spec, int out[3])
{
    /* ceil_div_grid((m, 1)) -- (total, tile) pairs with tile=1 each. */
    int totals[2];
    int tiles[2];

    (void)spec;
    if(out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    totals[0] = m;
    totals[1] = 1;
    tiles[0]  = 1;
    tiles[1]  = 1;
    return ckc_ceil_div_grid(totals, tiles, 2, out);
}

/* ===================================================================== *
 *  signature
 * ===================================================================== */

ckc_status_t ckc_add_rmsnorm2d_bf16_signature(ckc_arena_t* arena,
                                              const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                              const ckc_sig_entry_t** out_items,
                                              size_t* out_count)
{
    ckc_signature_builder_t sb;
    ckc_status_t st;

    if(arena == NULL || spec == NULL)
    {
        return CKC_ERR_VALUE;
    }
    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
    {
        return st;
    }
    ckc_signature_builder_ptr(&sb, "A", spec->dtype, NULL);
    ckc_signature_builder_ptr(&sb, "B", spec->dtype, NULL);
    ckc_signature_builder_ptr(&sb, "Gamma", spec->dtype, NULL);
    if(spec->save_residual)
    {
        ckc_signature_builder_ptr(&sb, "X", spec->dtype, NULL);
    }
    ckc_signature_builder_ptr(&sb, "Y", spec->dtype, NULL);
    ckc_signature_builder_scalar(&sb, "M", "i32");
    ckc_signature_builder_scalar(&sb, "N", "i32");
    ckc_signature_builder_scalar(&sb, "eps", "f32");
    return ckc_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  lower_to_llvm convenience -- owns and frees its own IRBuilder.
 * ===================================================================== */

ckc_status_t ckc_add_rmsnorm2d_bf16_lower_to_llvm(const ckc_add_rmsnorm2d_bf16_spec_t* spec,
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

    kernel = ckc_build_add_rmsnorm2d_bf16_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_add_rmsnorm2d_bf16 failed";
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
