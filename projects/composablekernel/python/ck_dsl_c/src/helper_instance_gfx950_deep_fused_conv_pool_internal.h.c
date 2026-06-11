/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_instance_gfx950_deep_fused_conv_pool_internal.h.c -- C99 port of the ten
 * PRIVATE shared-state + closure-phase symbols of the gfx950 (CDNA, wave64, MFMA
 * 32x32x16) arch shim over build_deep_fused_conv_pool
 *   (ck_dsl/instances/gfx950/deep_fused_conv_pool.py -> re-exported common
 *    ck_dsl/instances/common/deep_fused_conv_pool.py build_deep_fused_conv_pool,
 *    Python lines 1212-1401).
 *
 * SYMBOLS PORTED IN THIS TRANSLATION UNIT (exactly the requested set):
 *   - ckc_gfx950_dfcp_build_ctx_t                    (struct -- in the header)
 *   - ckc_gfx950_dfcp_build_ctx_init                 (Python prologue locals)
 *   - ckc_gfx950_dfcp_extra_params                   (extra_params -> W1 rsrc)
 *   - ckc_gfx950_dfcp_m_index_fn                     (m_index_fn -> M index)
 *   - ckc_gfx950_dfcp_a_mhw_index_fn                 (a_mhw_index_fn -> n,ho,wo)
 *   - ckc_gfx950_dfcp_setup_input_cache              (setup_input_cache)
 *   - ckc_gfx950_dfcp_setup_specialized_a_loader     (setup_specialized_a_loader)
 *   - ckc_gfx950_dfcp_load_a_tile_from_cache         (load_a_tile_from_cache)
 *   - ckc_gfx950_dfcp_load_a_tile_specialized        (load_a_tile_specialized)
 *   - ckc_gfx950_dfcp_load_a_operand_from_cache      (load_a_operand_from_cache)
 *   - ckc_gfx950_dfcp_epilogue_override              (epilogue_override)
 *
 * Each closure stages its per-callback args (grid / conv-managed resources) onto
 * the shared ctx so the body reads only the ctx, then delegates to the
 * family-agnostic common emit helpers (ckc_dfcp_* over ctx->common_spec). The
 * gfx950 closures carry no per-family branching in the numeric core; the MFMA op
 * resolved by the driver drives the common bodies. The builder call sequence is
 * byte-identical to the Python closures (and to the common port part-file
 * instance_deep_fused_conv_pool_build_entry_and_closures.c, aside from the
 * ctx->common_spec forwarding).
 *
 * gfx950 vs gfx1201 maxpool routing.
 *   gfx950 (MFMA 32x32) selects the intra-lane register-resident maxpool fast
 *   path (_maxpool_is_intra_lane / _emit_inline_maxpool_from_registers) when its
 *   exact 32x32 geometry holds; the WMMA intra-lane fast path is geometry-gated
 *   off; otherwise the layout-agnostic cshuffle-LDS gather + maxpool runs. This
 *   walk is byte-faithful to the common closure (Python lines 1358-1375).
 *
 * Public build driver + conv-builder trampolines are NOT part of this requested
 * set; they live in the gfx950 public-entry TU (when present) and are reached via
 * this internal header. STUB-TO-LINK: nothing in this set requires stubbing --
 * every body either emits IR via ckc_b_* / ckc_dfcp_* or stages ctx fields.
 */
#include "ckc/helper_instance_gfx950_deep_fused_conv_pool_internal.h.h"

#include <stddef.h>
#include <string.h> /* memset */

#include "ckc/ir.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err (unused-path parity) */
#include "ckc/helper_ck_dsl.instances.common.deep_fused_conv_pool.h"

/* ===================================================================== *
 * ckc_gfx950_dfcp_build_ctx_init -- the Python build_deep_fused_conv_pool
 * prologue, driven by the gfx950 (MFMA) op.
 *
 * Mirrors:
 *   ok, why = is_valid_spec(spec, arch); if not ok: raise ...   (driver gate)
 *   conv_spec = spec.conv_spec()
 *   op = _resolve_conv_op(conv_spec, "gfx950")
 *   + the override-routing flag derivation done at the build call tail.
 *
 * The validity gate + conv_spec/op resolution are performed by the public driver
 * (it owns the conv_spec/op storage); this init takes them as args and stages the
 * build-time-constant ctx fields, per the internal-header contract: normalise
 * arch -> "gfx950", common_spec = &spec->base, derive defer / deferred_epi, the
 * MFMA register-residency decision, and the A-load routing flags.
 * ===================================================================== */
ckc_status_t
ckc_gfx950_dfcp_build_ctx_init(ckc_gfx950_dfcp_build_ctx_t* ctx,
                               ckc_ir_builder_t* b,
                               const ckc_gfx950_deep_fused_conv_pool_spec_t* spec,
                               const char* arch,
                               const ckc_implicit_gemm_conv_spec_t* conv_spec,
                               const ckc_mma_op_t* op)
{
    const ckc_deep_fused_conv_pool_spec_t* cs;

    if (ctx == NULL || b == NULL || spec == NULL)
    {
        return CKC_ERR_VALUE;
    }

    memset(ctx, 0, sizeof(*ctx));

    /* (A) build-time constants ------------------------------------------ */
    ctx->b = b;
    ctx->spec = spec;
    /* common spec view of the gfx950 spec (== &spec->base); the closures forward
     * this to the family-agnostic common emit helpers. */
    ctx->common_spec = &spec->base;
    cs = ctx->common_spec;
    /* arch NULL => "gfx950" for this shim. */
    ctx->arch = (arch != NULL) ? arch : CKC_GFX950_DEEP_FUSED_CONV_POOL_ARCH;
    ctx->conv_spec = conv_spec;
    ctx->op = op;

    /* defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue)
     * deferred_epi = spec.conv1_epilogue if defer else None
     * (computed once here so each maxpool phase reads the single decision). */
    ctx->defer = ckc_dfcp_epilogue_is_pool_deferrable(&cs->conv1_epilogue);
    ctx->deferred_epi = ctx->defer ? &cs->conv1_epilogue : NULL;

    /* MFMA register-residency decision (gfx950 maxpool routing). The per-callback
     * grid is not known at prologue time; the epilogue phase stages the grid and
     * re-evaluates the predicate (_maxpool_is_intra_lane(spec, grid)) with it.
     * Staged here as false; re-derived in the epilogue dispatch over the live
     * grid. */
    ctx->use_mfma_register_maxpool = false;

    /* A-load routing (the build_implicit_gemm_conv override selection tail):
     *   use_input_cache  = cache_input_footprint or direct_conv0_from_input_cache
     *   use_specialized  = (not use_input_cache) and
     *                      _can_use_specialized_conv0_a_loader(spec)
     *   use_operand_ovr  = direct_conv0_from_input_cache */
    ctx->use_input_cache =
        cs->cache_input_footprint || cs->direct_conv0_from_input_cache;
    ctx->use_specialized =
        (!ctx->use_input_cache) &&
        ckc_dfcp_can_use_specialized_conv0_a_loader(cs);
    ctx->use_operand_ovr = cs->direct_conv0_from_input_cache;

    return CKC_OK;
}

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
ckc_value_t* ckc_gfx950_dfcp_extra_params(ckc_gfx950_dfcp_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b;
    const ckc_type_t* f16;
    const ckc_type_t* ptr_f16_global;
    ckc_param_opts_t opts;
    ckc_value_t* w1;
    ckc_value_t* w1_bytes;
    ckc_value_t* rsrc;

    if (ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }
    b = ctx->b;

    /* PtrType(F16, "global") */
    f16 = ckc_f16();
    ptr_f16_global = ckc_ptr_type(b, f16, "global");

    /* W1 = b.param("W1", ptr, noalias=True, readonly=True, align=16) */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    w1 = ckc_b_param(b, "W1", ptr_f16_global, &opts);

    /* W1_bytes = b.param("W1_bytes", I32) */
    w1_bytes = ckc_b_param(b, "W1_bytes", ckc_i32(), NULL);

    /* make_buffer_resource(b, W1, num_bytes=W1_bytes).rsrc
     *
     * Python's make_buffer_resource (helpers/tensor_view.py) builds the rsrc AND
     * a pre-bound zero soffset (soffset = b.const_i32(0)) before returning the
     * BufferResource; extra_params keeps only .rsrc. That discarded const_i32(0)
     * still consumes one build-time SSA counter (it is DCE'd before printing), so
     * every later numbered SSA name is +1 vs a port that omits it. Emit the
     * throwaway const here to stay byte-identical. */
    rsrc = ckc_b_buffer_rsrc(b, w1, w1_bytes);
    (void)ckc_b_const_i32(b, 0);

    ctx->w1_rsrc = rsrc;
    return rsrc;
}

/* ---- shared (ho, wo) decode used by m_index_fn / a_mhw_index_fn ----------
 * tile-local row -> (global_h, global_w). Strength-reduces div/mod by
 * conv_tile_w when power-of-2 (matches both Python closures verbatim). Reads
 * the common spec view of the gfx950 spec. */
static void ckc_gfx950_dfcp_decode_row_to_hw(
    ckc_ir_builder_t* b,
    const ckc_deep_fused_conv_pool_spec_t* spec,
    ckc_value_t* row,
    ckc_value_t** out_h,
    ckc_value_t** out_w)
{
    const ckc_fused_conv_pool_problem_t* p = &spec->problem;
    int conv_tile_w = spec->pool_tile_w * p->pool_stride_w;
    ckc_value_t* local_h;
    ckc_value_t* local_w;
    ckc_value_t* global_h;
    ckc_value_t* global_w;

    if (conv_tile_w > 0 && (conv_tile_w & (conv_tile_w - 1)) == 0)
    {
        /* shift = (conv_tile_w - 1).bit_length() */
        int shift = 0;
        int v = conv_tile_w - 1;
        while (v > 0)
        {
            ++shift;
            v >>= 1;
        }
        local_h = ckc_b_lshr(b, row, ckc_b_const_i32(b, shift));
        local_w = ckc_b_land(b, row, ckc_b_const_i32(b, conv_tile_w - 1));
    }
    else
    {
        ckc_value_t* c_conv_tile_w = ckc_b_const_i32(b, conv_tile_w);
        local_h = ckc_b_div(b, row, c_conv_tile_w);
        local_w = ckc_b_mod(b, row, c_conv_tile_w);
    }

    /* global_h = block_id_y()*(pool_tile_h*pool_stride_h) + local_h
     *
     * Python (b.mul(b.block_id_y(), b.const_i32(...))) evaluates its arguments
     * left-to-right, so block_id_y() takes its SSA counter slot BEFORE the const.
     * C function-call argument evaluation order is unspecified, which would swap
     * the two slots and shift every later numbered SSA name. Hoist block_id_y/_z
     * into temps first to pin the Python source-order. */
    {
        ckc_value_t* bid_y = ckc_b_block_id_y(b);
        global_h = ckc_b_add(
            b,
            ckc_b_mul(b, bid_y,
                      ckc_b_const_i32(b, spec->pool_tile_h * p->pool_stride_h)),
            local_h);
    }
    /* global_w = block_id_z()*(pool_tile_w*pool_stride_w) + local_w */
    {
        ckc_value_t* bid_z = ckc_b_block_id_z(b);
        global_w = ckc_b_add(
            b,
            ckc_b_mul(b, bid_z,
                      ckc_b_const_i32(b, spec->pool_tile_w * p->pool_stride_w)),
            local_w);
    }

    *out_h = global_h;
    *out_w = global_w;
}

/* ===================================================================== *
 * CLOSURE PHASE: m_index_fn(b, row, grid) -> global (ho, wo) flattened M index.
 *   return global_h * Wo + global_w
 * ===================================================================== */
ckc_value_t* ckc_gfx950_dfcp_m_index_fn(ckc_gfx950_dfcp_build_ctx_t* ctx,
                                        ckc_value_t* row,
                                        const ckc_warp_grid_t* grid)
{
    ckc_ir_builder_t* b;
    const ckc_conv_problem_t* c;
    ckc_value_t* global_h;
    ckc_value_t* global_w;

    (void)grid; /* Python `_grid` is unused in m_index_fn */
    if (ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return NULL;
    }
    b = ctx->b;
    c = &ctx->common_spec->problem.conv;

    ckc_gfx950_dfcp_decode_row_to_hw(b, ctx->common_spec, row, &global_h,
                                     &global_w);

    /* b.add(b.mul(global_h, b.const_i32(c.Wo)), global_w) */
    return ckc_b_add(
        b, ckc_b_mul(b, global_h, ckc_b_const_i32(b, ckc_conv_problem_wo(c))),
        global_w);
}

/* ===================================================================== *
 * CLOSURE PHASE: a_mhw_index_fn(b, row, grid) -> (n=0, global_h, global_w).
 *   Same (ho, wo) decode as m_index_fn, returned as separate coords. N==1 so
 *   n is constant 0.
 * ===================================================================== */
void ckc_gfx950_dfcp_a_mhw_index_fn(ckc_gfx950_dfcp_build_ctx_t* ctx,
                                    ckc_value_t* row,
                                    const ckc_warp_grid_t* grid,
                                    ckc_value_t** out_n,
                                    ckc_value_t** out_h,
                                    ckc_value_t** out_w)
{
    ckc_ir_builder_t* b;
    ckc_value_t* global_h;
    ckc_value_t* global_w;

    (void)grid;
    if (ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return;
    }
    b = ctx->b;

    ckc_gfx950_dfcp_decode_row_to_hw(b, ctx->common_spec, row, &global_h,
                                     &global_w);

    if (out_n != NULL)
    {
        *out_n = ckc_b_const_i32(b, 0);
    }
    if (out_h != NULL)
    {
        *out_h = global_h;
    }
    if (out_w != NULL)
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
ckc_value_t* ckc_gfx950_dfcp_setup_input_cache(
    ckc_gfx950_dfcp_build_ctx_t* ctx,
    const ckc_implicit_gemm_conv_spec_t* conv_spec_,
    const ckc_warp_grid_t* grid,
    ckc_value_t* a_rsrc)
{
    ckc_value_t* cache;

    if (ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return NULL;
    }
    /* stage per-callback scratch on the ctx so the body reads only the ctx */
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->a_rsrc = a_rsrc;

    cache = ckc_dfcp_setup_input_footprint_cache(ctx->b, ctx->common_spec,
                                                 a_rsrc, grid);
    ctx->input_cache = cache;
    return cache;
}

/* ===================================================================== *
 * CLOSURE PHASE: setup_specialized_a_loader(b, conv_spec_, grid, a_rsrc).
 *   return a_rsrc   (identity passthrough -- the specialized loader reads
 *                    global memory directly)
 * ===================================================================== */
ckc_value_t* ckc_gfx950_dfcp_setup_specialized_a_loader(
    ckc_gfx950_dfcp_build_ctx_t* ctx,
    const ckc_implicit_gemm_conv_spec_t* conv_spec_,
    const ckc_warp_grid_t* grid,
    ckc_value_t* a_rsrc)
{
    if (ctx == NULL)
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
void ckc_gfx950_dfcp_load_a_tile_from_cache(
    ckc_gfx950_dfcp_build_ctx_t* ctx,
    const ckc_implicit_gemm_conv_spec_t* conv_spec_,
    ckc_value_t* k_off,
    ckc_value_t* a_dst,
    const ckc_warp_grid_t* grid,
    ckc_value_t* cache)
{
    if (ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return;
    }
    /* Early-return when direct_conv0_from_input_cache (load is fused into the
     * operand override instead). */
    if (ctx->common_spec->direct_conv0_from_input_cache)
    {
        return;
    }
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->k_off = k_off;
    ctx->a_dst = a_dst;
    ctx->input_cache = cache;

    ckc_dfcp_load_conv0_a_tile_from_input_cache(ctx->b, ctx->common_spec,
                                                conv_spec_, k_off, a_dst, grid,
                                                cache);
}

/* ===================================================================== *
 * CLOSURE PHASE: load_a_tile_specialized(b, conv_spec_, k_off, a_dst, grid,
 *                                        a_rsrc)
 *   _load_conv0_a_tile_specialized(b, spec, conv_spec_, k_off, a_dst, grid,
 *                                  a_rsrc)
 * ===================================================================== */
void ckc_gfx950_dfcp_load_a_tile_specialized(
    ckc_gfx950_dfcp_build_ctx_t* ctx,
    const ckc_implicit_gemm_conv_spec_t* conv_spec_,
    ckc_value_t* k_off,
    ckc_value_t* a_dst,
    const ckc_warp_grid_t* grid,
    ckc_value_t* a_rsrc)
{
    if (ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
    {
        return;
    }
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->k_off = k_off;
    ctx->a_dst = a_dst;
    ctx->a_rsrc = a_rsrc;

    ckc_dfcp_load_conv0_a_tile_specialized(ctx->b, ctx->common_spec, conv_spec_,
                                           k_off, a_dst, grid, a_rsrc);
}

/* ===================================================================== *
 * CLOSURE PHASE: load_a_operand_from_cache(b, conv_spec_, row, k_off, col_base,
 *                                          frag_len, grid, cache) -> Value
 *   return _load_conv0_a_operand_from_input_cache(b, spec, row, k_off, col_base,
 *                                                 frag_len, cache)
 * ===================================================================== */
ckc_value_t* ckc_gfx950_dfcp_load_a_operand_from_cache(
    ckc_gfx950_dfcp_build_ctx_t* ctx,
    const ckc_implicit_gemm_conv_spec_t* conv_spec_,
    ckc_value_t* row,
    ckc_value_t* k_off,
    ckc_value_t* col_base,
    int frag_len,
    const ckc_warp_grid_t* grid,
    ckc_value_t* cache)
{
    if (ctx == NULL || ctx->b == NULL || ctx->common_spec == NULL)
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

    return ckc_dfcp_load_conv0_a_operand_from_input_cache(
        ctx->b, ctx->common_spec, row, k_off, col_base, frag_len, cache);
}

/* ===================================================================== *
 * CLOSURE PHASE: epilogue_override(b, conv_spec_, accs, grid, y_rsrc, w1_rsrc)
 *
 * Walks (Python order, deep_fused_conv_pool.py:1334-1375), driven by the gfx950
 * (MFMA 32x32x16) op:
 *   c_smem  = _stage_accumulators_to_cshuffle_lds(b, op, accs, grid, sync=False)
 *   w1_smem = _load_conv1_weights_to_lds(b, spec, w1_rsrc, grid, sync=False)
 *   b.sync()                                   # single merged barrier
 *   defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue)  (staged ctx->defer)
 *   conv1_accs = _emit_conv1_1x1(..., defer_epilogue=defer)
 *   deferred_epi = spec.conv1_epilogue if defer else None
 *   if _maxpool_is_intra_lane(spec, grid):                     # MFMA fast path
 *       _emit_inline_maxpool_from_registers(..., epilogue=deferred_epi)
 *   elif _maxpool_is_intra_lane_wmma(spec, grid, op):          # WMMA fast path
 *       _emit_wmma_maxpool_from_registers(..., op, epilogue=deferred_epi)
 *   else:
 *       conv1_smem = _stage_accumulators_to_cshuffle_lds(b, op, conv1_accs, grid)
 *       _emit_inline_maxpool_from_cshuffle(..., epilogue=deferred_epi)
 * ===================================================================== */
void ckc_gfx950_dfcp_epilogue_override(ckc_gfx950_dfcp_build_ctx_t* ctx,
                                       const ckc_implicit_gemm_conv_spec_t* conv_spec_,
                                       ckc_value_t* const* accs,
                                       size_t num_accs,
                                       const ckc_warp_grid_t* grid,
                                       ckc_value_t* y_rsrc,
                                       ckc_value_t* w1_rsrc)
{
    ckc_ir_builder_t* b;
    const ckc_deep_fused_conv_pool_spec_t* spec; /* common spec view (&spec->base) */
    const ckc_mma_op_t* op;                      /* MFMA 32x32x16 op               */
    bool defer;
    const ckc_conv_acc_epilogue_t* deferred_epi;
    ckc_status_t st;
    size_t i;

    if (ctx == NULL || ctx->b == NULL)
    {
        return;
    }
    b = ctx->b;
    spec = ctx->common_spec; /* the family-agnostic emit helpers know only this */
    op = ctx->op;            /* resolved MFMA op (wave64, m=n=32, k=16)         */

    /* Stage per-callback scratch onto the ctx so the body reads only the ctx. */
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->y_rsrc = y_rsrc;
    ctx->w1_rsrc = w1_rsrc;
    ctx->num_conv0_accs = 0;
    for (i = 0; i < num_accs && i < (size_t)CKC_GFX950_DFCP_MAX_ACCS; ++i)
    {
        ctx->conv0_accs[i] = accs[i];
    }
    ctx->num_conv0_accs = (num_accs < (size_t)CKC_GFX950_DFCP_MAX_ACCS)
                              ? num_accs
                              : (size_t)CKC_GFX950_DFCP_MAX_ACCS;

    /* Barrier-merge: the conv0 cshuffle stage (writes DeepFusionC_smem) and the
     * W1 load (writes W1_smem) target disjoint LDS tiles, and the conv1 MMA below
     * reads both. Emit each producer without its own barrier and gate the
     * consumer on a single block-wide barrier; this also lets the W1 global loads
     * overlap the conv0 cshuffle LDS stores. */
    ctx->c_smem = ckc_dfcp_stage_accumulators_to_cshuffle_lds(
        b, op, accs, num_accs, grid, /*sync=*/false);
    ctx->w1_smem = ckc_dfcp_load_conv1_weights_to_lds(b, spec, w1_rsrc, grid,
                                                      /*sync=*/false);
    ckc_b_sync(b);

    /* VALU opt: ReLU/bias/clamp/(scale>=0) are monotonic, so the conv1 epilogue
     * commutes with maxpool. Defer it past the pool to apply once per pooled
     * pixel instead of per conv1 acc element (~4x fewer fmax). The decision is the
     * one ckc_gfx950_dfcp_build_ctx_init computed from spec.conv1_epilogue. */
    defer = ctx->defer;

    st = ckc_dfcp_emit_conv1_1x1(b, spec, conv_spec_, op, ctx->c_smem,
                                 ctx->w1_smem, grid, /*defer_epilogue=*/defer,
                                 ctx->conv1_accs, (size_t)CKC_GFX950_DFCP_MAX_ACCS,
                                 &ctx->num_conv1_accs);
    if (st != CKC_OK)
    {
        /* error already routed through b by the emit helper */
        return;
    }

    deferred_epi = defer ? &spec->conv1_epilogue : NULL;
    ctx->deferred_epi = deferred_epi;

    /* MFMA register-residency decision -- re-derived over the live per-callback
     * grid (the prologue cannot see the grid). For gfx950 the 32x32 geometry
     * makes this the selected fast path. */
    ctx->use_mfma_register_maxpool = ckc_dfcp_maxpool_is_intra_lane(spec, grid);

    if (ctx->use_mfma_register_maxpool)
    {
        /* MFMA-32x32 register-resident fast path. Each lane's vec<16> conv1
         * accumulator already holds the 4 pool windows it owns (intra-lane, no
         * shuffle), so reduce straight to global output. */
        ckc_dfcp_emit_inline_maxpool_from_registers(
            b, spec, ctx->conv1_accs, ctx->num_conv1_accs, y_rsrc, grid,
            deferred_epi);
    }
    else if (ckc_dfcp_maxpool_is_intra_lane_wmma(spec, grid, op))
    {
        /* RDNA4 analogue -- geometry-gated off for the MFMA warp_tile 32x32, so
         * this branch is never taken here; preserved to keep the walk
         * byte-faithful to the common closure. */
        ckc_dfcp_emit_wmma_maxpool_from_registers(b, spec, ctx->conv1_accs,
                                                  ctx->num_conv1_accs, y_rsrc,
                                                  grid, op, deferred_epi);
    }
    else
    {
        /* Generic layout-agnostic cshuffle-LDS gather + maxpool: stage the conv1
         * accs to LDS (with its own barrier) and reduce from there. */
        ckc_value_t* conv1_smem = ckc_dfcp_stage_accumulators_to_cshuffle_lds(
            b, op, ctx->conv1_accs, ctx->num_conv1_accs, grid, /*sync=*/true);
        ctx->conv1_smem = conv1_smem;
        ckc_dfcp_emit_inline_maxpool_from_cshuffle(b, spec, conv1_smem, y_rsrc,
                                                   grid, deferred_epi);
    }
}
