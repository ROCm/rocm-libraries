// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1201_deep_fused_conv_pool_conv0_aload_closures.c -- the conv0
 * A-load closure phases + coord callbacks of the chunked C99 port of the gfx1201
 * (RDNA4, wave32, WMMA 16x16x16) arch shim over build_deep_fused_conv_pool
 *   (rocke/instances/gfx1201/deep_fused_conv_pool.py -> re-exported common
 *    rocke/instances/common/deep_fused_conv_pool.py build_deep_fused_conv_pool,
 *    Python lines 1212-1401; the conv0 A-load nested closures
 *    1222-1332).
 *
 * SCOPE OF THIS PART-FILE (one bucket of the chunked gfx1201 port):
 *   - rocke_gfx1201_dfcp_extra_params           (extra_params -> W1 buffer rsrc)
 *   - rocke_gfx1201_dfcp_m_index_fn             (m_index_fn -> flattened M index)
 *   - rocke_gfx1201_dfcp_a_mhw_index_fn         (a_mhw_index_fn -> (n,ho,wo))
 *   - rocke_gfx1201_dfcp_setup_input_cache      (setup_input_cache)
 *   - rocke_gfx1201_dfcp_setup_specialized_a_loader (setup_specialized_a_loader)
 *   - rocke_gfx1201_dfcp_load_a_tile_from_cache (load_a_tile_from_cache)
 *   - rocke_gfx1201_dfcp_load_a_tile_specialized(load_a_tile_specialized)
 *   - rocke_gfx1201_dfcp_load_a_operand_from_cache (load_a_operand_from_cache)
 *
 * Each phase stages its per-callback args (grid / conv-managed resources) onto
 * the shared ctx so the body reads only the ctx, then delegates to the
 * family-agnostic common emit helpers (rocke_dfcp_*) over ctx->common_spec -- the
 * gfx1201 closures carry no per-family branching in the numeric core; the WMMA
 * op resolved by the driver drives the common bodies. Byte-identical builder
 * call sequence to the Python closures (and to the common port part-file
 * instance_deep_fused_conv_pool_build_entry_and_closures.c, with which these are
 * byte-faithful aside from the ctx->common_spec forwarding).
 *
 * Peer phases (ctx init, epilogue_override, the public driver + trampolines)
 * live in sibling gfx1201 part-files and are reached only via the internal
 * header; this TU implements ONLY the conv0 A-load closures + coord callbacks.
 */
#include "rocke/instance_gfx1201_deep_fused_conv_pool.h"
#include "rocke/instance_gfx1201_deep_fused_conv_pool_internal.h"

#include <stddef.h>
#include <string.h> /* memset */

#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h"
#include "rocke/ir.h"

/* ===================================================================== *
 * CLOSURE PHASE: extra_params(b) -> W1 buffer resource.
 *
 *   W1       = b.param("W1", PtrType(F16,"global"), noalias=True,
 *                      readonly=True, align=16)
 *   W1_bytes = b.param("W1_bytes", I32)
 *   return make_buffer_resource(b, W1, num_bytes=W1_bytes).rsrc
 *
 * Stores the rsrc in ctx->w1_rsrc (captured by epilogue_override) and returns
 * it. Identical to the common closure -- the W1 loader is family-agnostic. */
rocke_value_t* rocke_gfx1201_dfcp_extra_params(rocke_gfx1201_dfcp_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b;
    const rocke_type_t* f16;
    const rocke_type_t* ptr_f16_global;
    rocke_param_opts_t opts;
    rocke_value_t* w1;
    rocke_value_t* w1_bytes;
    rocke_value_t* rsrc;

    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }
    b = ctx->b;

    /* PtrType(F16, "global") */
    f16 = rocke_f16();
    ptr_f16_global = rocke_ptr_type(b, f16, "global");

    /* W1 = b.param("W1", ptr, noalias=True, readonly=True, align=16) */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    w1 = rocke_b_param(b, "W1", ptr_f16_global, &opts);

    /* W1_bytes = b.param("W1_bytes", I32) */
    w1_bytes = rocke_b_param(b, "W1_bytes", rocke_i32(), NULL);

    /* make_buffer_resource(b, W1, num_bytes=W1_bytes).rsrc
     *
     * Python's make_buffer_resource (helpers/tensor_view.py) builds the rsrc AND
     * a pre-bound zero soffset (soffset = b.const_i32(0)) before returning the
     * BufferResource; extra_params keeps only .rsrc. That discarded const_i32(0)
     * still consumes one build-time SSA counter (it is DCE'd before printing), so
     * every later numbered SSA name is +1 vs a port that omits it. Emit the
     * throwaway const here to stay byte-identical (otherwise @A_smem22 vs the
     * reference @A_smem23, and the whole IR is shifted by -1). */
    rsrc = rocke_b_buffer_rsrc(b, w1, w1_bytes);
    (void)rocke_b_const_i32(b, 0);

    ctx->w1_rsrc = rsrc;
    return rsrc;
}

/* ---- shared (ho, wo) decode used by m_index_fn / a_mhw_index_fn ----------
 * tile-local row -> (global_h, global_w). Strength-reduces div/mod by
 * conv_tile_w when power-of-2 (matches both Python closures verbatim). Reads
 * the common spec view of the gfx1201 spec. */
static void rocke_gfx1201_dfcp_decode_row_to_hw(rocke_ir_builder_t* b,
                                                const rocke_deep_fused_conv_pool_spec_t* spec,
                                                rocke_value_t* row,
                                                rocke_value_t** out_h,
                                                rocke_value_t** out_w)
{
    const rocke_fused_conv_pool_problem_t* p = &spec->problem;
    int conv_tile_w = spec->pool_tile_w * p->pool_stride_w;
    rocke_value_t* local_h;
    rocke_value_t* local_w;
    rocke_value_t* global_h;
    rocke_value_t* global_w;

    if(conv_tile_w > 0 && (conv_tile_w & (conv_tile_w - 1)) == 0)
    {
        /* shift = (conv_tile_w - 1).bit_length() */
        int shift = 0;
        int v = conv_tile_w - 1;
        while(v > 0)
        {
            ++shift;
            v >>= 1;
        }
        local_h = rocke_b_lshr(b, row, rocke_b_const_i32(b, shift));
        local_w = rocke_b_land(b, row, rocke_b_const_i32(b, conv_tile_w - 1));
    }
    else
    {
        rocke_value_t* c_conv_tile_w = rocke_b_const_i32(b, conv_tile_w);
        local_h = rocke_b_div(b, row, c_conv_tile_w);
        local_w = rocke_b_mod(b, row, c_conv_tile_w);
    }

    /* global_h = block_id_y()*(pool_tile_h*pool_stride_h) + local_h
     *
     * Python (b.mul(b.block_id_y(), b.const_i32(...))) evaluates its arguments
     * left-to-right, so block_id_y() takes its SSA counter slot BEFORE the const.
     * C function-call argument evaluation order is unspecified (GCC evaluates
     * right-to-left here), which would swap the two slots and shift every later
     * numbered SSA name. Hoist block_id_y/_z into temps first to pin the
     * Python source-order. */
    {
        rocke_value_t* bid_y = rocke_b_block_id_y(b);
        global_h = rocke_b_add(
            b,
            rocke_b_mul(b, bid_y, rocke_b_const_i32(b, spec->pool_tile_h * p->pool_stride_h)),
            local_h);
    }
    /* global_w = block_id_z()*(pool_tile_w*pool_stride_w) + local_w */
    {
        rocke_value_t* bid_z = rocke_b_block_id_z(b);
        global_w = rocke_b_add(
            b,
            rocke_b_mul(b, bid_z, rocke_b_const_i32(b, spec->pool_tile_w * p->pool_stride_w)),
            local_w);
    }

    *out_h = global_h;
    *out_w = global_w;
}

/* ===================================================================== *
 * CLOSURE PHASE: m_index_fn(b, row, grid) -> global (ho, wo) flattened M index.
 *   return global_h * Wo + global_w
 * ===================================================================== */
rocke_value_t* rocke_gfx1201_dfcp_m_index_fn(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                             rocke_value_t* row,
                                             const rocke_warp_grid_t* grid)
{
    rocke_ir_builder_t* b;
    const rocke_conv_problem_t* c;
    rocke_value_t* global_h;
    rocke_value_t* global_w;

    (void)grid; /* Python `_grid` is unused in m_index_fn */
    if(ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return NULL;
    }
    b = ctx->b;
    c = &ctx->common_spec->problem.conv;

    rocke_gfx1201_dfcp_decode_row_to_hw(b, ctx->common_spec, row, &global_h, &global_w);

    /* b.add(b.mul(global_h, b.const_i32(c.Wo)), global_w) */
    return rocke_b_add(
        b, rocke_b_mul(b, global_h, rocke_b_const_i32(b, rocke_conv_problem_wo(c))), global_w);
}

/* ===================================================================== *
 * CLOSURE PHASE: a_mhw_index_fn(b, row, grid) -> (n=0, global_h, global_w).
 *   Same (ho, wo) decode as m_index_fn, returned as separate coords. N==1 so
 *   n is constant 0.
 * ===================================================================== */
void rocke_gfx1201_dfcp_a_mhw_index_fn(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                       rocke_value_t* row,
                                       const rocke_warp_grid_t* grid,
                                       rocke_value_t** out_n,
                                       rocke_value_t** out_h,
                                       rocke_value_t** out_w)
{
    rocke_ir_builder_t* b;
    rocke_value_t* global_h;
    rocke_value_t* global_w;

    (void)grid;
    if(ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return;
    }
    b = ctx->b;

    rocke_gfx1201_dfcp_decode_row_to_hw(b, ctx->common_spec, row, &global_h, &global_w);

    if(out_n != NULL)
    {
        *out_n = rocke_b_const_i32(b, 0);
    }
    if(out_h != NULL)
    {
        *out_h = global_h;
    }
    if(out_w != NULL)
    {
        *out_w = global_w;
    }
}

/* ===================================================================== *
 * CLOSURE PHASE: setup_input_cache(b, conv_spec_, grid, a_rsrc) -> cache.
 *   return _setup_input_footprint_cache(b, spec, a_rsrc, grid)
 * Delegates to the common helper over ctx->common_spec; stages + returns the
 * cache (ctx->input_cache).
 * ===================================================================== */
rocke_value_t*
    rocke_gfx1201_dfcp_setup_input_cache(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                         const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                         const rocke_warp_grid_t* grid,
                                         rocke_value_t* a_rsrc)
{
    rocke_value_t* cache;

    if(ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return NULL;
    }
    /* stage per-callback scratch on the ctx so the body reads only the ctx */
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->a_rsrc = a_rsrc;

    cache = rocke_dfcp_setup_input_footprint_cache(ctx->b, ctx->common_spec, a_rsrc, grid);
    ctx->input_cache = cache;
    return cache;
}

/* ===================================================================== *
 * CLOSURE PHASE: setup_specialized_a_loader(b, conv_spec_, grid, a_rsrc).
 *   return a_rsrc   (identity passthrough -- the specialized loader reads
 *                    global memory directly)
 * ===================================================================== */
rocke_value_t*
    rocke_gfx1201_dfcp_setup_specialized_a_loader(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                                  const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                                  const rocke_warp_grid_t* grid,
                                                  rocke_value_t* a_rsrc)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->a_rsrc = a_rsrc;
    ctx->input_cache = a_rsrc; /* the specialized loader reads global directly */
    return a_rsrc;
}

/* ===================================================================== *
 * CLOSURE PHASE: load_a_tile_from_cache(b, conv_spec_, k_off, a_dst, grid, cache)
 *   if spec.direct_conv0_from_input_cache: return        (no-op)
 *   _load_conv0_a_tile_from_input_cache(b, spec, conv_spec_, k_off, a_dst, grid,
 *                                       cache)
 * ===================================================================== */
void rocke_gfx1201_dfcp_load_a_tile_from_cache(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                               const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                               rocke_value_t* k_off,
                                               rocke_value_t* a_dst,
                                               const rocke_warp_grid_t* grid,
                                               rocke_value_t* cache)
{
    if(ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return;
    }
    /* Early-return when direct_conv0_from_input_cache (load is fused into the
     * operand override instead). */
    if(ctx->common_spec->direct_conv0_from_input_cache)
    {
        return;
    }
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->k_off = k_off;
    ctx->a_dst = a_dst;
    ctx->input_cache = cache;

    rocke_dfcp_load_conv0_a_tile_from_input_cache(
        ctx->b, ctx->common_spec, conv_spec_, k_off, a_dst, grid, cache);
}

/* ===================================================================== *
 * CLOSURE PHASE: load_a_tile_specialized(b, conv_spec_, k_off, a_dst, grid,
 *                                        a_rsrc)
 *   _load_conv0_a_tile_specialized(b, spec, conv_spec_, k_off, a_dst, grid,
 *                                  a_rsrc)
 * ===================================================================== */
void rocke_gfx1201_dfcp_load_a_tile_specialized(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                                const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                                rocke_value_t* k_off,
                                                rocke_value_t* a_dst,
                                                const rocke_warp_grid_t* grid,
                                                rocke_value_t* a_rsrc)
{
    if(ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return;
    }
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->k_off = k_off;
    ctx->a_dst = a_dst;
    ctx->a_rsrc = a_rsrc;

    rocke_dfcp_load_conv0_a_tile_specialized(
        ctx->b, ctx->common_spec, conv_spec_, k_off, a_dst, grid, a_rsrc);
}

/* ===================================================================== *
 * CLOSURE PHASE: load_a_operand_from_cache(b, conv_spec_, row, k_off, col_base,
 *                                          frag_len, grid, cache) -> Value
 *   return _load_conv0_a_operand_from_input_cache(b, spec, row, k_off, col_base,
 *                                                 frag_len, cache)
 * ===================================================================== */
rocke_value_t*
    rocke_gfx1201_dfcp_load_a_operand_from_cache(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                                 const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                                 rocke_value_t* row,
                                                 rocke_value_t* k_off,
                                                 rocke_value_t* col_base,
                                                 int frag_len,
                                                 const rocke_warp_grid_t* grid,
                                                 rocke_value_t* cache)
{
    if(ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return NULL;
    }
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->row = row;
    ctx->k_off = k_off;
    ctx->col_base = col_base;
    ctx->frag_len = frag_len;
    ctx->input_cache = cache;

    return rocke_dfcp_load_conv0_a_operand_from_input_cache(
        ctx->b, ctx->common_spec, row, k_off, col_base, frag_len, cache);
}
