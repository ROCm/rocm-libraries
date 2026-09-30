// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_fused_moe_silu.c -- C99 port of the build_moe_silu_mul and
 * build_moe_silu_mul_packed phase functions
 * (rocke/instances/common/fused_moe.py, lines 449-573 + 587-708).
 *
 * Scope of THIS translation unit (both SwiGLU activation builders over the
 * shared rocke_moe_stream_ctx_t):
 *   rocke_moe_silu_mul_prologue         -- lines 491-523
 *   rocke_moe_silu_mul_body_scalar      -- lines 526-538 (VEC==1 fallback)
 *   rocke_moe_silu_mul_body_tile        -- lines 540-573 (CK-Tile dist path)
 *   rocke_moe_silu_mul_packed_prologue  -- lines 612-648
 *   rocke_moe_silu_mul_packed_body_scalar -- lines 658-670 (VEC==1 fallback)
 *   rocke_moe_silu_mul_packed_body_tile -- lines 674-708 (CK-Tile dist path)
 *
 * The builder-call sequence is byte-identical to the Python source so a lowered
 * .ll diffs clean. Peers resolved via their headers: rocke_moe_effective_vec,
 * rocke_moe_chunk_distribution, rocke_moe_silu_mul_f32 (internal header),
 * rocke_fused_moe_is_valid_spec, rocke_fused_moe_spec_elems_per_thread_inter,
 * rocke_io_ir_type, rocke_b_load_scalar_as_f32, rocke_b_store_scalar_from_f32.
 *
 * TILE PATH NOTE.
 *   The distribution-aware tile.load(distribution=, ps=) / store(out_dt, ps=) /
 *   TileDistribution.iterate_ys() surface is not a standalone C helper yet (the
 *   same gap instance_elementwise.c documents). The fixed single-X / single-P /
 *   single-Y (BS, VEC) chunk distribution these two builders exercise is
 *   reproduced here as static load_tile / store_tile helpers that mirror the
 *   Python builder-call order exactly:
 *     load_tile : x = calculate_x(ys=[const_i32(0)], ps=[[tid]]);
 *                 scalars = window.load_vec_as_f32(*x, n=VEC)
 *     iterate_ys: a plain C for(y<VEC) over the single stride-1 Y dim
 *     store_tile: x = calculate_x(...); casts=cast_f32_to; vec_pack; store_vec
 *   The packed variant reuses ONE GateUp view for both the gate slab (origin
 *   gate_base) and the up slab (origin up_base), I_DIM apart.
 */

#include <stddef.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.distribution.h" /* tile distribution surface  */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_io_ir_type, *_scalar_* */
#include "rocke/helper_rocke.helpers.tensor_view.h" /* views / windows            */
#include "rocke/instance_fused_moe.h"
#include "rocke/instance_fused_moe_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err              */

/* ===================================================================== *
 *  Distribution-driven load / store for the fixed (BS, VEC) chunk tile.
 *
 *  Reproduces the Python load_tile / store_tile builder-call order for the
 *  single-Y, single-P, single-X distribution _chunk_distribution(BS, VEC)
 *  produces. The per-thread register tile holds exactly VEC f32 scalars
 *  (storage[0..VEC)). Identical machinery to instance_elementwise.c's
 *  rocke_ew_load_tile / rocke_ew_store_tile.
 * ===================================================================== */

/* load_tile(window, distribution, ps=[[tid]]):
 *   ys      = [b.const_i32(0)]                  (single Y access, y_base==(0,))
 *   x_coords= distribution.calculate_x(ys=ys, ps=[[tid]])
 *   v       = window.load_vec(*x_coords, n=VEC)
 *   dt[k]   = cast_to_f32(vec_extract(v, k))    (load_vec_as_f32 inline)
 * out_storage capacity >= vec. Returns 1 on success, 0 on builder failure. */
static int rocke_moe_silu_load_tile(rocke_ir_builder_t* b,
                                    const rocke_tile_window_t* window,
                                    const rocke_tile_distribution_t* dist,
                                    rocke_value_t* tid,
                                    int vec,
                                    rocke_value_t** out_storage)
{
    rocke_value_t* ys[1];
    rocke_value_t* ps_row[1];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* x_coords[1];
    rocke_value_t* loaded;
    int k;

    ys[0] = rocke_b_const_i32(b, 0);
    ps_row[0] = tid;
    ps[0] = ps_row;
    ps_counts[0] = 1;

    if(!rocke_tile_distribution_calculate_x(b, dist, ys, 1, ps, ps_counts, 1, x_coords, 1))
    {
        return 0;
    }

    /* window.load_vec_as_f32(*x_coords, n=vec) -- vec is >= 2 here, so the n==1
     * scalar branch never triggers. */
    loaded = rocke_tile_window_load_vec(b, window, x_coords, 1, vec);
    for(k = 0; k < vec; ++k)
    {
        out_storage[k] = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, loaded, k));
    }
    return rocke_ir_builder_ok(b) ? 1 : 0;
}

/* store_tile(window, distributed, ps=[[tid]]):
 *   x_coords = calculate_x(ys=[const_i32(0)], ps=[[tid]])
 *   casts[k] = cast_f32_to(storage[k], dtype)
 *   packed   = vec_pack(casts, dtype)
 *   window.store_vec(*x_coords, packed, n=vec)
 */
static void rocke_moe_silu_store_tile(rocke_ir_builder_t* b,
                                      const rocke_tile_window_t* window,
                                      const rocke_tile_distribution_t* dist,
                                      rocke_value_t* tid,
                                      int vec,
                                      rocke_value_t** storage)
{
    rocke_value_t* ys[1];
    rocke_value_t* ps_row[1];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* x_coords[1];
    rocke_value_t* casts[ROCKE_TV_MAX_VEC];
    rocke_value_t* packed;
    const rocke_type_t* dtype;
    int k;

    ys[0] = rocke_b_const_i32(b, 0);
    ps_row[0] = tid;
    ps[0] = ps_row;
    ps_counts[0] = 1;

    if(!rocke_tile_distribution_calculate_x(b, dist, ys, 1, ps, ps_counts, 1, x_coords, 1))
    {
        return;
    }

    dtype = rocke_tile_window_dtype(window);
    for(k = 0; k < vec; ++k)
    {
        casts[k] = rocke_b_cast_f32_to(b, storage[k], dtype);
    }
    packed = rocke_b_vec_pack(b, casts, vec, dtype);
    rocke_tile_window_store_vec(b, window, x_coords, 1, packed, vec);
}

/* ===================================================================== *
 *  SILU_MUL PROLOGUE  (build_moe_silu_mul, lines 491-523)
 * ===================================================================== */
bool rocke_moe_silu_mul_prologue(rocke_moe_stream_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL || ctx->spec == NULL)
    {
        return false;
    }

    rocke_ir_builder_t* b = ctx->b;
    const rocke_fused_moe_spec_t* spec = ctx->spec;

    /* ok, why = is_valid_spec(spec); if not ok: raise ValueError(...) */
    {
        char why[ROCKE_ERR_MSG_CAP];
        if(!rocke_fused_moe_is_valid_spec(spec, why, sizeof(why)))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid fused_moe spec: %s", why);
            return false;
        }
    }

    /* I_DIM = spec.intermediate; BS = spec.block_size;
     * EPT = spec.elems_per_thread_inter;
     * VEC = _effective_vec(spec.vec, BS, I_DIM); dtype = spec.dtype */
    ctx->kind = ROCKE_MOE_STREAM_SILU_MUL;
    ctx->N = spec->intermediate; /* I_DIM            */
    ctx->BS = spec->block_size; /* BS               */
    ctx->EPT = rocke_fused_moe_spec_elems_per_thread_inter(spec);
    ctx->VEC = rocke_moe_effective_vec(spec->vec, ctx->BS, ctx->N);
    ctx->dtype = spec->dtype;

    /* b.kernel.attrs["max_workgroup_size"] = BS (name seeded by *_new entry). */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->BS);

    /* ty = io_ir_type(dtype) */
    ctx->ty = rocke_io_ir_type(ctx->dtype);

    /* GateOut = b.param("GateOut", PtrType(ty,"global"),
     *                   noalias=True, readonly=True, align=16) */
    {
        rocke_param_opts_t opts = {0};
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx->GateOut = rocke_b_param(b, "GateOut", rocke_ptr_type(b, ctx->ty, "global"), &opts);
    }
    /* UpOut = b.param("UpOut", PtrType(ty,"global"),
     *                 noalias=True, readonly=True, align=16) */
    {
        rocke_param_opts_t opts = {0};
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx->UpOut = rocke_b_param(b, "UpOut", rocke_ptr_type(b, ctx->ty, "global"), &opts);
    }
    /* Hidden = b.param("Hidden", PtrType(ty,"global"),
     *                  noalias=True, writeonly=True, align=16) */
    {
        rocke_param_opts_t opts = {0};
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx->Hidden = rocke_b_param(b, "Hidden", rocke_ptr_type(b, ctx->ty, "global"), &opts);
    }
    /* _total_pairs = b.param("total_pairs", I32)  # ABI */
    ctx->p_total_pairs = rocke_b_param(b, "total_pairs", rocke_i32(), NULL);
    /* _inter = b.param("intermediate", I32)  # ABI */
    ctx->p_inter = rocke_b_param(b, "intermediate", rocke_i32(), NULL);

    /* bid = b.block_id_x(); tid = b.thread_id_x() */
    ctx->bid = rocke_b_block_id_x(b);
    ctx->tid = rocke_b_thread_id_x(b);
    /* row_base = b.mul(bid, b.const_i32(I_DIM)) */
    ctx->row_base = rocke_b_mul(b, ctx->bid, rocke_b_const_i32(b, ctx->N));

    /* c_neg_log2e = b.const_f32(-1.4426950408889634); one_f32 = b.const_f32(1.0) */
    ctx->c_neg_log2e = rocke_b_const_f32(b, -1.4426950408889634);
    ctx->one_f32 = rocke_b_const_f32(b, 1.0);
    /* c_vec = b.const_i32(VEC) */
    ctx->c_vec = rocke_b_const_i32(b, ctx->VEC);

    /* chunks = EPT // VEC */
    ctx->chunks = ctx->EPT / ctx->VEC;

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  SILU_MUL SCALAR BODY  (VEC==1 fallback, lines 526-538)
 * ===================================================================== */
rocke_kernel_def_t* rocke_moe_silu_mul_body_scalar(rocke_moe_stream_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }

    rocke_ir_builder_t* b = ctx->b;
    const int BS = ctx->BS;
    const int VEC = ctx->VEC;
    const char* dtype = ctx->dtype;
    int k;

    for(k = 0; k < ctx->chunks; ++k)
    {
        /* i_col = b.add(b.const_i32(k*BS*VEC), b.mul(tid, c_vec))
         * Python evaluates the two arguments left-to-right, so the const
         * is created before the mul; bind each to a temp in that order so
         * the SSA value-id counter matches (C arg eval is right-to-left). */
        rocke_value_t* i_col_c = rocke_b_const_i32(b, (int64_t)k * BS * VEC);
        rocke_value_t* i_col_m = rocke_b_mul(b, ctx->tid, ctx->c_vec);
        rocke_value_t* i_col = rocke_b_add(b, i_col_c, i_col_m);
        /* off = b.add(row_base, i_col) */
        rocke_value_t* off = rocke_b_add(b, ctx->row_base, i_col);
        /* g = load_scalar_as_f32(b, GateOut, off, dtype=dtype) */
        rocke_value_t* g = rocke_b_load_scalar_as_f32(b, ctx->GateOut, off, dtype);
        /* u = load_scalar_as_f32(b, UpOut, off, dtype=dtype) */
        rocke_value_t* u = rocke_b_load_scalar_as_f32(b, ctx->UpOut, off, dtype);
        /* h = _silu_mul_f32(b, g, u, one_f32=one_f32, c_neg_log2e=c_neg_log2e) */
        rocke_value_t* h = rocke_moe_silu_mul_f32(b, g, u, ctx->one_f32, ctx->c_neg_log2e);
        /* store_scalar_from_f32(b, Hidden, off, h, dtype=dtype) */
        rocke_b_store_scalar_from_f32(b, ctx->Hidden, off, h, dtype);
    }

    /* return b.kernel */
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

/* ===================================================================== *
 *  SILU_MUL TILE BODY  (VEC>1 CK-Tile distribution path, lines 540-573)
 * ===================================================================== */
rocke_kernel_def_t* rocke_moe_silu_mul_body_tile(rocke_moe_stream_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }

    rocke_ir_builder_t* b = ctx->b;
    const int BS = ctx->BS;
    const int VEC = ctx->VEC;
    int shape1[1];
    int k;

    /* distribution = _chunk_distribution(BS, VEC) */
    ctx->distribution = rocke_moe_chunk_distribution(b, BS, VEC);
    if(ctx->distribution == NULL)
    {
        return NULL;
    }
    /* chunk_elems = BS * VEC */
    ctx->chunk_elems = BS * VEC;

    /* gate_view = make_global_view(GateOut, shape=(chunk_elems,), dtype=ty)
     * up_view   = make_global_view(UpOut,   shape=(chunk_elems,), dtype=ty)
     * out_view  = make_global_view(Hidden,  shape=(chunk_elems,), dtype=ty) */
    shape1[0] = ctx->chunk_elems;
    if(rocke_make_global_view(&ctx->gate_view, ctx->GateOut, shape1, 1, ctx->ty, NULL) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_global_view(GateOut) failed");
        return NULL;
    }
    if(rocke_make_global_view(&ctx->up_view, ctx->UpOut, shape1, 1, ctx->ty, NULL) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_global_view(UpOut) failed");
        return NULL;
    }
    if(rocke_make_global_view(&ctx->out_view, ctx->Hidden, shape1, 1, ctx->ty, NULL) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_global_view(Hidden) failed");
        return NULL;
    }

    /* ps = [[tid]] -- carried as ctx->ps_tid for parity with the contract. */
    ctx->ps_tid = ctx->tid;

    for(k = 0; k < ctx->chunks; ++k)
    {
        rocke_value_t* col_off;
        rocke_value_t* origin[1];
        int lengths1[1];
        rocke_tile_window_t gate_tile, up_tile, out_tile;
        rocke_value_t* g_dt[ROCKE_TV_MAX_VEC];
        rocke_value_t* u_dt[ROCKE_TV_MAX_VEC];
        rocke_value_t* out_dt[ROCKE_TV_MAX_VEC];
        int y;

        /* chunk_origin = (b.add(row_base, b.const_i32(k*BS*VEC)),) */
        col_off = rocke_b_const_i32(b, (int64_t)k * BS * VEC);
        origin[0] = rocke_b_add(b, ctx->row_base, col_off);
        lengths1[0] = ctx->chunk_elems;

        /* gate_tile / up_tile / out_tile = make_tile_window(view, (chunk_elems,),
         * origin=chunk_origin) -- all anchored at the same chunk origin. */
        if(rocke_make_tile_window(&gate_tile, &ctx->gate_view, lengths1, origin, 1) != ROCKE_OK
           || rocke_make_tile_window(&up_tile, &ctx->up_view, lengths1, origin, 1) != ROCKE_OK
           || rocke_make_tile_window(&out_tile, &ctx->out_view, lengths1, origin, 1) != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_tile_window(silu_mul) failed");
            return NULL;
        }

        /* g_dt = gate_tile.load(b, distribution=distribution, ps=ps) */
        if(!rocke_moe_silu_load_tile(b, &gate_tile, ctx->distribution, ctx->tid, VEC, g_dt))
        {
            return NULL;
        }
        /* u_dt = up_tile.load(b, distribution=distribution, ps=ps) */
        if(!rocke_moe_silu_load_tile(b, &up_tile, ctx->distribution, ctx->tid, VEC, u_dt))
        {
            return NULL;
        }
        /* out_dt = make_static_distributed_tensor(distribution, dtype=ty)
         * for y in distribution.iterate_ys():
         *     out_dt.set(y, _silu_mul_f32(b, g_dt.get(y), u_dt.get(y), ...)) */
        for(y = 0; y < VEC; ++y)
        {
            out_dt[y] = rocke_moe_silu_mul_f32(b, g_dt[y], u_dt[y], ctx->one_f32, ctx->c_neg_log2e);
        }
        /* out_tile.store(b, out_dt, ps=ps) */
        rocke_moe_silu_store_tile(b, &out_tile, ctx->distribution, ctx->tid, VEC, out_dt);
        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
    }

    /* return b.kernel */
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

/* ===================================================================== *
 *  SILU_MUL_PACKED PROLOGUE  (build_moe_silu_mul_packed, lines 612-648)
 * ===================================================================== */
bool rocke_moe_silu_mul_packed_prologue(rocke_moe_stream_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL || ctx->spec == NULL)
    {
        return false;
    }

    rocke_ir_builder_t* b = ctx->b;
    const rocke_fused_moe_spec_t* spec = ctx->spec;

    /* ok, why = is_valid_spec(spec); if not ok: raise ValueError(...) */
    {
        char why[ROCKE_ERR_MSG_CAP];
        if(!rocke_fused_moe_is_valid_spec(spec, why, sizeof(why)))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid fused_moe spec: %s", why);
            return false;
        }
    }

    /* I_DIM = spec.intermediate; BS = spec.block_size;
     * EPT = spec.elems_per_thread_inter;
     * VEC = _effective_vec(spec.vec, BS, I_DIM); dtype = spec.dtype */
    ctx->kind = ROCKE_MOE_STREAM_SILU_MUL_PACKED;
    ctx->N = spec->intermediate; /* I_DIM            */
    ctx->BS = spec->block_size; /* BS               */
    ctx->EPT = rocke_fused_moe_spec_elems_per_thread_inter(spec);
    ctx->VEC = rocke_moe_effective_vec(spec->vec, ctx->BS, ctx->N);
    ctx->dtype = spec->dtype;

    /* b.kernel.attrs["max_workgroup_size"] = BS (name seeded by *_new entry). */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->BS);

    /* ty = io_ir_type(dtype) */
    ctx->ty = rocke_io_ir_type(ctx->dtype);

    /* GateUp = b.param("GateUp", PtrType(ty,"global"),
     *                  noalias=True, readonly=True, align=16) */
    {
        rocke_param_opts_t opts = {0};
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx->GateUp = rocke_b_param(b, "GateUp", rocke_ptr_type(b, ctx->ty, "global"), &opts);
    }
    /* Hidden = b.param("Hidden", PtrType(ty,"global"),
     *                  noalias=True, writeonly=True, align=16) */
    {
        rocke_param_opts_t opts = {0};
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx->Hidden = rocke_b_param(b, "Hidden", rocke_ptr_type(b, ctx->ty, "global"), &opts);
    }
    /* _total_pairs = b.param("total_pairs", I32)  # ABI */
    ctx->p_total_pairs = rocke_b_param(b, "total_pairs", rocke_i32(), NULL);
    /* _inter = b.param("intermediate", I32)  # ABI */
    ctx->p_inter = rocke_b_param(b, "intermediate", rocke_i32(), NULL);

    /* bid = b.block_id_x(); tid = b.thread_id_x() */
    ctx->bid = rocke_b_block_id_x(b);
    ctx->tid = rocke_b_thread_id_x(b);

    /* two_i = b.const_i32(2*I_DIM); i_const = b.const_i32(I_DIM)
     * gate_base = b.mul(bid, two_i)
     * up_base   = b.add(gate_base, i_const)
     * out_base  = b.mul(bid, i_const) */
    {
        rocke_value_t* two_i = rocke_b_const_i32(b, (int64_t)2 * ctx->N);
        rocke_value_t* i_const = rocke_b_const_i32(b, ctx->N);
        ctx->gate_base = rocke_b_mul(b, ctx->bid, two_i);
        ctx->up_base = rocke_b_add(b, ctx->gate_base, i_const);
        ctx->out_base = rocke_b_mul(b, ctx->bid, i_const);
    }

    /* c_neg_log2e = b.const_f32(-1.4426950408889634); one_f32 = b.const_f32(1.0) */
    ctx->c_neg_log2e = rocke_b_const_f32(b, -1.4426950408889634);
    ctx->one_f32 = rocke_b_const_f32(b, 1.0);
    /* c_vec = b.const_i32(VEC) */
    ctx->c_vec = rocke_b_const_i32(b, ctx->VEC);

    /* chunks = EPT // VEC */
    ctx->chunks = ctx->EPT / ctx->VEC;

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  SILU_MUL_PACKED SCALAR BODY  (VEC==1 fallback, lines 658-670)
 * ===================================================================== */
rocke_kernel_def_t* rocke_moe_silu_mul_packed_body_scalar(rocke_moe_stream_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }

    rocke_ir_builder_t* b = ctx->b;
    const int BS = ctx->BS;
    const int VEC = ctx->VEC;
    const char* dtype = ctx->dtype;
    int k;

    for(k = 0; k < ctx->chunks; ++k)
    {
        /* i_col = b.add(b.const_i32(k*BS*VEC), b.mul(tid, c_vec))
         * Python evaluates the two arguments left-to-right, so the const
         * is created before the mul; bind each to a temp in that order so
         * the SSA value-id counter matches (C arg eval is right-to-left). */
        rocke_value_t* i_col_c = rocke_b_const_i32(b, (int64_t)k * BS * VEC);
        rocke_value_t* i_col_m = rocke_b_mul(b, ctx->tid, ctx->c_vec);
        rocke_value_t* i_col = rocke_b_add(b, i_col_c, i_col_m);
        /* g_off = b.add(gate_base, i_col) */
        rocke_value_t* g_off = rocke_b_add(b, ctx->gate_base, i_col);
        /* u_off = b.add(up_base, i_col) */
        rocke_value_t* u_off = rocke_b_add(b, ctx->up_base, i_col);
        /* o_off = b.add(out_base, i_col) */
        rocke_value_t* o_off = rocke_b_add(b, ctx->out_base, i_col);
        /* g = load_scalar_as_f32(b, GateUp, g_off, dtype=dtype) */
        rocke_value_t* g = rocke_b_load_scalar_as_f32(b, ctx->GateUp, g_off, dtype);
        /* u = load_scalar_as_f32(b, GateUp, u_off, dtype=dtype) */
        rocke_value_t* u = rocke_b_load_scalar_as_f32(b, ctx->GateUp, u_off, dtype);
        /* h = _silu_mul_f32(b, g, u, one_f32=one_f32, c_neg_log2e=c_neg_log2e) */
        rocke_value_t* h = rocke_moe_silu_mul_f32(b, g, u, ctx->one_f32, ctx->c_neg_log2e);
        /* store_scalar_from_f32(b, Hidden, o_off, h, dtype=dtype) */
        rocke_b_store_scalar_from_f32(b, ctx->Hidden, o_off, h, dtype);
    }

    /* return b.kernel */
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

/* ===================================================================== *
 *  SILU_MUL_PACKED TILE BODY  (VEC>1 CK-Tile distribution path, 674-708)
 * ===================================================================== */
rocke_kernel_def_t* rocke_moe_silu_mul_packed_body_tile(rocke_moe_stream_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }

    rocke_ir_builder_t* b = ctx->b;
    const int BS = ctx->BS;
    const int VEC = ctx->VEC;
    int shape1[1];
    int k;

    /* distribution = _chunk_distribution(BS, VEC) */
    ctx->distribution = rocke_moe_chunk_distribution(b, BS, VEC);
    if(ctx->distribution == NULL)
    {
        return NULL;
    }
    /* chunk_elems = BS * VEC */
    ctx->chunk_elems = BS * VEC;

    /* gateup_view = make_global_view(GateUp, shape=(chunk_elems,), dtype=ty)
     * out_view    = make_global_view(Hidden, shape=(chunk_elems,), dtype=ty)
     * (the packed variant reuses ONE GateUp view for both gate + up slabs;
     * ctx->gate_view holds the shared GateUp view.) */
    shape1[0] = ctx->chunk_elems;
    if(rocke_make_global_view(&ctx->gate_view, ctx->GateUp, shape1, 1, ctx->ty, NULL) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_global_view(GateUp) failed");
        return NULL;
    }
    if(rocke_make_global_view(&ctx->out_view, ctx->Hidden, shape1, 1, ctx->ty, NULL) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_global_view(Hidden) failed");
        return NULL;
    }

    /* ps = [[tid]] -- carried as ctx->ps_tid for parity with the contract. */
    ctx->ps_tid = ctx->tid;

    for(k = 0; k < ctx->chunks; ++k)
    {
        rocke_value_t* col_off;
        rocke_value_t* gate_origin[1];
        rocke_value_t* up_origin[1];
        rocke_value_t* out_origin[1];
        int lengths1[1];
        rocke_tile_window_t gate_tile, up_tile, out_tile;
        rocke_value_t* g_dt[ROCKE_TV_MAX_VEC];
        rocke_value_t* u_dt[ROCKE_TV_MAX_VEC];
        rocke_value_t* out_dt[ROCKE_TV_MAX_VEC];
        int y;

        /* col_off = b.const_i32(k*BS*VEC)
         * gate_origin = (b.add(gate_base, col_off),)
         * up_origin   = (b.add(up_base,  col_off),)
         * out_origin  = (b.add(out_base, col_off),) */
        col_off = rocke_b_const_i32(b, (int64_t)k * BS * VEC);
        gate_origin[0] = rocke_b_add(b, ctx->gate_base, col_off);
        up_origin[0] = rocke_b_add(b, ctx->up_base, col_off);
        out_origin[0] = rocke_b_add(b, ctx->out_base, col_off);
        lengths1[0] = ctx->chunk_elems;

        /* gate_tile = make_tile_window(gateup_view, (chunk_elems,), gate_origin)
         * up_tile   = make_tile_window(gateup_view, (chunk_elems,), up_origin)
         *   (both index the SAME GateUp view at the two G1U1 slab origins)
         * out_tile  = make_tile_window(out_view, (chunk_elems,), out_origin) */
        if(rocke_make_tile_window(&gate_tile, &ctx->gate_view, lengths1, gate_origin, 1) != ROCKE_OK
           || rocke_make_tile_window(&up_tile, &ctx->gate_view, lengths1, up_origin, 1) != ROCKE_OK
           || rocke_make_tile_window(&out_tile, &ctx->out_view, lengths1, out_origin, 1)
                  != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_tile_window(silu_mul_packed) failed");
            return NULL;
        }

        /* g_dt = gate_tile.load(b, distribution=distribution, ps=ps) */
        if(!rocke_moe_silu_load_tile(b, &gate_tile, ctx->distribution, ctx->tid, VEC, g_dt))
        {
            return NULL;
        }
        /* u_dt = up_tile.load(b, distribution=distribution, ps=ps) */
        if(!rocke_moe_silu_load_tile(b, &up_tile, ctx->distribution, ctx->tid, VEC, u_dt))
        {
            return NULL;
        }
        /* out_dt = make_static_distributed_tensor(distribution, dtype=ty)
         * for y in distribution.iterate_ys():
         *     out_dt.set(y, _silu_mul_f32(b, g_dt.get(y), u_dt.get(y), ...)) */
        for(y = 0; y < VEC; ++y)
        {
            out_dt[y] = rocke_moe_silu_mul_f32(b, g_dt[y], u_dt[y], ctx->one_f32, ctx->c_neg_log2e);
        }
        /* out_tile.store(b, out_dt, ps=ps) */
        rocke_moe_silu_store_tile(b, &out_tile, ctx->distribution, ctx->tid, VEC, out_dt);
        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
    }

    /* return b.kernel */
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}
