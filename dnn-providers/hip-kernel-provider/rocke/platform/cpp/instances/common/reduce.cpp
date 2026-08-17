// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_reduce.c -- C99 port of rocke/instances/common/reduce.py.
 *
 * Row-wise reduction kernel (M,N) -> (M,1): sum/max/min/mean/prod per row.
 * Byte-identical builder-call sequence vs the Python build_reduce2d.
 *
 * Stage 1 (thread): tree_reduce f32 fold of each vec-chunk, joined onto acc.
 * Stage 2 (cross-thread): wave-aligned sum/max -> reduce-distribution +
 *   block_tile_reduce_sync; wave-aligned min/prod -> wave-XOR prologue;
 *   non-wave -> full LDS tree.
 * Stage 3 (lane 0): scalar store of the f32 result demoted to dtype.
 */

#include "rocke/instance_reduce.h"

#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.distribution.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.sweep.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* The reduce-distribution helpers (make_reduce_tile_distribution_encoding,
 * make_static_distributed_tensor, block_tile_reduce_sync) and the
 * rocke_static_distributed_tensor_t type are now provided by the distribution
 * port (helper_rocke.helpers.distribution.{h,c}), included above. */

/* ===================================================================== *
 *  f32 identity constants (reduce.py module-level).
 * ===================================================================== */
#define ROCKE_REDUCE_NEG_INF_F32 (-3.4028234663852886e38)
#define ROCKE_REDUCE_POS_INF_F32 (3.4028234663852886e38)

/* ===================================================================== *
 *  Spec defaults / properties.
 * ===================================================================== */
rocke_reduce2d_spec_t rocke_reduce2d_spec_default(void)
{
    rocke_reduce2d_spec_t s;
    s.n_per_block = 0; /* required; caller sets */
    s.op = "sum";
    s.block_size = 256;
    s.vec = 4;
    s.dtype = "f16";
    s.wave_size = 64;
    s.name = "rocke_reduce2d";
    return s;
}

int rocke_reduce2d_elems_per_thread(const rocke_reduce2d_spec_t* spec)
{
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return spec->n_per_block / spec->block_size;
}

int rocke_reduce2d_num_warps(const rocke_reduce2d_spec_t* spec)
{
    if(spec == NULL || spec->wave_size == 0)
    {
        return 0;
    }
    return spec->block_size / spec->wave_size;
}

/* ===================================================================== *
 *  kernel_name():
 *    kernel_name_join(name, op, dtype, f"N{n}", f"b{bs}", f"v{vec}")
 * ===================================================================== */
rocke_status_t
    rocke_reduce2d_kernel_name(const rocke_reduce2d_spec_t* spec, char* out, size_t out_cap)
{
    char nbuf[32];
    char bbuf[32];
    char vbuf[32];
    const char* parts[5];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(nbuf, sizeof(nbuf), "N%d", spec->n_per_block);
    snprintf(bbuf, sizeof(bbuf), "b%d", spec->block_size);
    snprintf(vbuf, sizeof(vbuf), "v%d", spec->vec);

    parts[0] = spec->op;
    parts[1] = spec->dtype;
    parts[2] = nbuf;
    parts[3] = bbuf;
    parts[4] = vbuf;

    return rocke_kernel_name_join(spec->name,
                                  parts,
                                  5,
                                  NULL,
                                  NULL,
                                  0, /* no flags */
                                  out,
                                  out_cap,
                                  NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec):
 *    op in (...); validate_io(IOSpecRule(dtype, block_size, vec, n_per_block))
 * ===================================================================== */
static int rocke_reduce2d_op_ok(const char* op)
{
    return op != NULL
           && (strcmp(op, "sum") == 0 || strcmp(op, "max") == 0 || strcmp(op, "min") == 0
               || strcmp(op, "mean") == 0 || strcmp(op, "prod") == 0);
}

bool rocke_reduce2d_is_valid_spec(const rocke_reduce2d_spec_t* spec,
                                  char* reason,
                                  size_t reason_cap)
{
    rocke_io_spec_rule_t rule;
    rocke_arena_t arena;
    const char* why = NULL;
    int ok;

    if(spec == NULL)
    {
        return false;
    }

    /* if spec.op not in (...): return False, f"unsupported op {spec.op!r}" */
    if(!rocke_reduce2d_op_ok(spec->op))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unsupported op '%s'", spec->op ? spec->op : "(null)");
        }
        return false;
    }

    /* return validate_io(IOSpecRule(dtype, block_size, vec, n_per_block)) */
    rocke_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec);
    rule.n_per_block_set = 1;
    rule.n_per_block = spec->n_per_block;

    rocke_arena_init(&arena, 0);
    ok = rocke_validate_io(&arena, &rule, &why);
    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "%s", why ? why : (ok ? "ok" : "invalid"));
    }
    rocke_arena_destroy(&arena);
    return ok != 0;
}

/* ===================================================================== *
 *  _combine_scalar(b, combine, a, c): one f32 reduction step.
 *  combine here is a rocke_reduce_combine_t (sum/max/min/prod).
 * ===================================================================== */
static rocke_value_t* rocke_reduce2d_combine_scalar(rocke_ir_builder_t* b,
                                                    rocke_reduce_combine_t combine,
                                                    rocke_value_t* a,
                                                    rocke_value_t* c)
{
    switch(combine)
    {
    case ROCKE_REDUCE_SUM:
        return rocke_b_fadd(b, a, c);
    case ROCKE_REDUCE_MAX:
        return rocke_b_fmax(b, a, c);
    case ROCKE_REDUCE_MIN:
        return rocke_b_fmin(b, a, c);
    case ROCKE_REDUCE_PROD:
        return rocke_b_fmul(b, a, c);
    default:
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "unsupported combine");
        return NULL;
    }
}

/* ===================================================================== *
 *  _make_row_reduce_distribution(spec):
 *    Hs = ((1, nwarp, tpw, 1),)
 *    Ps2RHs_major = ((1,), (1,)); Ps2RHs_minor = ((1,), (2,))
 *    Ys2RHs_major = (1, 1);       Ys2RHs_minor = (0, 3)
 *    reduce_enc = make_reduce_tile_distribution_encoding(encoding, [0])
 *    return make_static_tile_distribution(reduce_enc)
 * ===================================================================== */
static const rocke_tile_distribution_t*
    rocke_reduce2d_make_row_reduce_distribution(rocke_ir_builder_t* b,
                                                const rocke_reduce2d_spec_t* spec)
{
    int nwarp = spec->block_size / spec->wave_size;
    int tpw = spec->wave_size;

    int h0[4];
    rocke_h_row_t Hs[1];

    int p0_major[1];
    int p0_minor[1];
    int p1_major[1];
    int p1_minor[1];
    rocke_p_seq_t Ps[2];

    int ys_major[2];
    int ys_minor[2];

    const rocke_tile_distribution_encoding_t* enc;
    rocke_tile_distribution_encoding_t* reduce_enc;
    int reduce_dim_xs[1];

    /* Hs = ((1, nwarp, tpw, 1),) */
    h0[0] = 1;
    h0[1] = nwarp;
    h0[2] = tpw;
    h0[3] = 1;
    Hs[0].levels = h0;
    Hs[0].count = 4;

    /* Ps2RHs_major = ((1,), (1,)); Ps2RHs_minor = ((1,), (2,)) */
    p0_major[0] = 1;
    p0_minor[0] = 1;
    p1_major[0] = 1;
    p1_minor[0] = 2;
    Ps[0].major = p0_major;
    Ps[0].minor = p0_minor;
    Ps[0].count = 1;
    Ps[1].major = p1_major;
    Ps[1].minor = p1_minor;
    Ps[1].count = 1;

    /* Ys2RHs_major = (1, 1); Ys2RHs_minor = (0, 3) */
    ys_major[0] = 1;
    ys_major[1] = 1;
    ys_minor[0] = 0;
    ys_minor[1] = 3;

    enc = rocke_make_tile_distribution_encoding(b,
                                                NULL,
                                                0, /* Rs (empty) */
                                                Hs,
                                                1,
                                                Ps,
                                                2,
                                                ys_major,
                                                ys_minor,
                                                2);
    if(enc == NULL)
    {
        return NULL;
    }

    /* reduce_enc = make_reduce_tile_distribution_encoding(encoding, [0]) */
    reduce_dim_xs[0] = 0;
    reduce_enc = rocke_make_reduce_tile_distribution_encoding(b, enc, reduce_dim_xs, 1);
    if(reduce_enc == NULL)
    {
        return NULL;
    }

    return rocke_make_static_tile_distribution(b, reduce_enc);
}

/* ===================================================================== *
 *  sweep_row_chunks body closure: per-chunk f32 tree fold joined onto acc.
 * ===================================================================== */
typedef struct rocke_reduce2d_body_ctx
{
    rocke_reduce_combine_t combine;
    rocke_value_t* acc; /* the `nonlocal acc` cell */
} rocke_reduce2d_body_ctx_t;

/* tree_reduce combine callback: forwards to _combine_scalar with the op. */
static rocke_value_t* rocke_reduce2d_tree_combine_cb(rocke_ir_builder_t* b,
                                                     rocke_value_t* a,
                                                     rocke_value_t* c,
                                                     void* user)
{
    rocke_reduce2d_body_ctx_t* ctx = (rocke_reduce2d_body_ctx_t*)user;
    return rocke_reduce2d_combine_scalar(b, ctx->combine, a, c);
}

/* body(_n_off, x_scalars):
 *   chunk_partial = tree_reduce(b, lambda a,c: _combine_scalar(...), list(x))
 *   acc = _combine_scalar(b, combine, acc, chunk_partial)
 */
static void rocke_reduce2d_sweep_body(rocke_ir_builder_t* b,
                                      rocke_value_t* n_off,
                                      rocke_value_t* const* x_scalars,
                                      int vec,
                                      void* user)
{
    rocke_reduce2d_body_ctx_t* ctx = (rocke_reduce2d_body_ctx_t*)user;
    rocke_value_t* chunk_partial;

    (void)n_off; /* Python: _n_off unused */

    chunk_partial = rocke_tree_reduce(b, rocke_reduce2d_tree_combine_cb, ctx, x_scalars, vec);
    ctx->acc = rocke_reduce2d_combine_scalar(b, ctx->combine, ctx->acc, chunk_partial);
}

/* ===================================================================== *
 *  build_reduce2d(spec) -- the build entry.
 * ===================================================================== */
rocke_kernel_def_t*
    rocke_build_reduce2d(rocke_ir_builder_t* b, const rocke_reduce2d_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char reason[ROCKE_ERR_MSG_CAP];
        const rocke_type_t* io_ty;
        int BS;
        int VEC;
        int N;

        rocke_value_t* X;
        rocke_value_t* Y;
        rocke_value_t* tid;
        rocke_value_t* row;

        rocke_tensor_view_t x_view;
        rocke_tile_window_t x_tile;
        rocke_value_t* lds;

        rocke_reduce_combine_t combine;
        rocke_value_t* acc;
        rocke_reduce2d_body_ctx_t bctx;
        rocke_value_t* total;

        (void)arch; /* Python build_reduce2d takes no arch. */

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }

        /* ok, why = is_valid_spec(spec); if not ok: raise ValueError(...) */
        if(!rocke_reduce2d_is_valid_spec(spec, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid reduce2d spec: %s", reason);
            return NULL;
        }

        /* io_ty = io_ir_type(spec.dtype) */
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

        /* X = b.param("X", PtrType(io_ty,"global"), noalias, readonly, align16)
         * Y = b.param("Y", PtrType(io_ty,"global"), noalias, writeonly, align16)
         * M = b.param("M", I32); _ = b.param("N", I32) */
        {
            rocke_param_opts_t opts;
            const rocke_type_t* ptr_elem = rocke_ptr_type(b, io_ty, "global");

            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.readonly = true;
            opts.readonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            X = rocke_b_param(b, "X", ptr_elem, &opts);

            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.writeonly = true;
            opts.writeonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            Y = rocke_b_param(b, "Y", ptr_elem, &opts);

            (void)rocke_b_param(b, "M", rocke_i32(), NULL);
            (void)rocke_b_param(b, "N", rocke_i32(), NULL);
        }

        /* tid = b.thread_id_x(); row = b.block_id_x() */
        tid = rocke_b_thread_id_x(b);
        row = rocke_b_block_id_x(b);

        /* x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
         *   == make_global_view(X, (1, N), io_ty) with packed strides. */
        {
            int shape2[2];
            shape2[0] = 1;
            shape2[1] = N;
            if(rocke_make_global_view(&x_view, X, shape2, 2, io_ty, NULL) != ROCKE_OK)
            {
                (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "reduce2d: make_global_view failed");
                return NULL;
            }
        }

        /* x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, const_i32(0))) */
        {
            int lens2[2];
            rocke_value_t* origin2[2];
            lens2[0] = 1;
            lens2[1] = N;
            origin2[0] = row;
            origin2[1] = rocke_b_const_i32(b, 0);
            if(rocke_make_tile_window(&x_tile, &x_view, lens2, origin2, 2) != ROCKE_OK)
            {
                (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "reduce2d: make_tile_window failed");
                return NULL;
            }
        }

        /* lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base
         *   == smem_alloc(F32, [BS], "lds_red"); .base is the smem token. */
        {
            int lds_shape[1];
            lds_shape[0] = BS;
            lds = rocke_b_smem_alloc(b, rocke_f32(), lds_shape, 1, "lds_red");
        }

        /* Pick f32 identity + combiner per op. */
        if(strcmp(spec->op, "sum") == 0 || strcmp(spec->op, "mean") == 0)
        {
            acc = rocke_b_const_f32(b, 0.0);
            combine = ROCKE_REDUCE_SUM;
        }
        else if(strcmp(spec->op, "max") == 0)
        {
            acc = rocke_b_const_f32(b, ROCKE_REDUCE_NEG_INF_F32);
            combine = ROCKE_REDUCE_MAX;
        }
        else if(strcmp(spec->op, "min") == 0)
        {
            acc = rocke_b_const_f32(b, ROCKE_REDUCE_POS_INF_F32);
            combine = ROCKE_REDUCE_MIN;
        }
        else if(strcmp(spec->op, "prod") == 0)
        {
            acc = rocke_b_const_f32(b, 1.0);
            combine = ROCKE_REDUCE_PROD;
        }
        else
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "unsupported reduce op '%s'", spec->op);
            return NULL;
        }

        /* sweep_row_chunks(b, x_tile, tid=tid, block_size=BS, vec=VEC,
         *                  elems_per_thread=spec.elems_per_thread, body=body)
         * The body threads acc through the bctx cell (Python `nonlocal acc`). */
        bctx.combine = combine;
        bctx.acc = acc;
        (void)rocke_sweep_row_chunks(b,
                                     &x_tile,
                                     tid,
                                     BS,
                                     VEC,
                                     rocke_reduce2d_elems_per_thread(spec),
                                     NULL, /* row=None (origin already pinned) */
                                     rocke_reduce2d_sweep_body,
                                     &bctx,
                                     false /* cache=False */);
        acc = bctx.acc;

        /* Cross-thread reduction. */
        if(spec->block_size % spec->wave_size == 0
           && (combine == ROCKE_REDUCE_SUM || combine == ROCKE_REDUCE_MAX))
        {
            /* red_dist = _make_row_reduce_distribution(spec)
             * reduced  = make_static_distributed_tensor(red_dist, dtype=F32)
             * reduced.storage[0] = acc
             * block_tile_reduce_sync(b, reduced, combine, lds_buf=lds, tid, wave_size)
             * total = reduced.storage[0] */
            const rocke_tile_distribution_t* red_dist;
            rocke_static_distributed_tensor_t* reduced;

            red_dist = rocke_reduce2d_make_row_reduce_distribution(b, spec);
            if(red_dist == NULL)
            {
                return NULL;
            }
            reduced = rocke_make_static_distributed_tensor(b, red_dist, rocke_f32());
            if(reduced == NULL || reduced->num_storage < 1)
            {
                (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "reduce2d: empty distributed tensor");
                return NULL;
            }
            reduced->storage[0] = acc;
            rocke_block_tile_reduce_sync(b, reduced, combine, lds, tid, spec->wave_size);
            total = reduced->storage[0];
        }
        else if(spec->block_size % spec->wave_size == 0)
        {
            /* min / prod: hand-built wave-XOR prologue. */
            total = rocke_block_lds_reduce_with_wave_prologue(
                b, acc, lds, tid, spec->block_size, combine, spec->wave_size);
        }
        else
        {
            /* full LDS tree. */
            total = rocke_block_lds_reduce(b, acc, lds, tid, BS, combine);
        }

        /* if spec.op == "mean": total = total * rcp(const_f32(float(N))) */
        if(strcmp(spec->op, "mean") == 0)
        {
            total = rocke_b_fmul(b, total, rocke_b_rcp(b, rocke_b_const_f32(b, (double)N)));
        }

        /* with b.scf_if(b.cmp_eq(tid, const_i32(0))):
         *     store_scalar_from_f32(b, Y, row, total, dtype=spec.dtype) */
        {
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
            rocke_b_region_enter(b, iff.then_region);
            rocke_b_store_scalar_from_f32(b, Y, row, total, spec->dtype);
            rocke_b_region_leave(b);
        }

        /* return b.kernel  (Python build_reduce2d returns b.kernel WITHOUT an
         * explicit cf.return; the trailing ret void is appended at lowering time,
         * so do NOT emit rocke_b_ret here -- matches the Python IR byte-for-byte). */

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

/* ===================================================================== *
 *  rocke_build_reduce2d_new -- init builder with spec.kernel_name() then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_reduce2d_new(rocke_ir_builder_t* b,
                                             const rocke_reduce2d_spec_t* spec,
                                             const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_reduce2d_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_reduce2d(b, spec, arch);
    });
}

/* ===================================================================== *
 *  reduce2d_grid(m, spec) -> ceil_div_grid((m, 1)) == (m, 1, 1).
 * ===================================================================== */
rocke_status_t rocke_reduce2d_grid(int m, const rocke_reduce2d_spec_t* spec, int out[3])
{
    int totals[1];
    int tiles[1];
    (void)spec;
    if(out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    totals[0] = m;
    tiles[0] = 1;
    return rocke_ceil_div_grid(totals, tiles, 1, out);
}

/* ===================================================================== *
 *  reduce2d_signature(spec):
 *    SignatureBuilder().ptr("X", dtype).ptr("Y", dtype)
 *                      .scalar("M","i32").scalar("N","i32").build()
 * ===================================================================== */
rocke_status_t rocke_reduce2d_signature(rocke_arena_t* arena,
                                        const rocke_reduce2d_spec_t* spec,
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
    rocke_signature_builder_ptr(&sb, "X", spec->dtype, NULL);
    rocke_signature_builder_ptr(&sb, "Y", spec->dtype, NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  rocke_reduce2d_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_reduce2d_lower_to_llvm(const rocke_reduce2d_spec_t* spec,
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

    kernel = rocke_build_reduce2d_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_reduce2d failed";
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
