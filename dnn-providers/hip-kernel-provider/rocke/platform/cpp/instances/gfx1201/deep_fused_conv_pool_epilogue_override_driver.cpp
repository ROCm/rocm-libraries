// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1201_deep_fused_conv_pool_epilogue_override_driver.c -- C99 port of
 * the gfx1201 (RDNA4, wave32, WMMA 16x16x16) `epilogue_override` closure from
 * build_deep_fused_conv_pool (re-exported common builder; common Python source
 * rocke/instances/common/deep_fused_conv_pool.py lines 1334-1375), driven here
 * by the WMMA-resolved op.
 *
 * SCOPE OF THIS TRANSLATION UNIT.
 *   Exactly one phase function: rocke_gfx1201_dfcp_epilogue_override -- the fused
 *   write-back walked in Python order. It:
 *     - stages the conv0 accumulators to the cshuffle LDS tile (sync=False) and
 *       loads W1 into a disjoint LDS tile (sync=False), then issues a single
 *       merged block-wide barrier so the two disjoint producers overlap and the
 *       conv1 consumer gates on one barrier (barrier-merge optimization);
 *     - emits the conv1 1x1 GEMM, deferring the conv1 epilogue past the maxpool
 *       when ctx->defer (the pool-deferral decision the prologue computed once);
 *     - routes the maxpool: the WMMA register-resident fast path
 *       (rocke_dfcp_emit_wmma_maxpool_from_registers) when
 *       ctx->use_wmma_register_maxpool, else the generic cshuffle-LDS gather
 *       maxpool (stage conv1 accs to LDS + rocke_dfcp_emit_inline_maxpool_from_
 *       cshuffle).
 *
 *   The MFMA-32x32 intra-lane register-resident maxpool fast path
 *   (_maxpool_is_intra_lane / _emit_inline_maxpool_from_registers) is
 *   geometry-gated off for the WMMA warp_tile 16x16 here, so it is preserved as
 *   the first branch (byte-faithful to the Python walk) but never selected; the
 *   WMMA routing then uses the staged ctx->use_wmma_register_maxpool decision
 *   (== _maxpool_is_intra_lane_wmma(common_spec, grid, op), pre-derived by
 *   rocke_gfx1201_dfcp_build_ctx_init).
 *
 * The per-callback scratch (conv_spec_, accs, grid, y_rsrc, w1_rsrc) is staged
 * onto the shared ctx so the body reads only the ctx, then the phases are walked
 * in Python order. Builder calls are byte-identical to the common closure driven
 * by the WMMA-resolved op (op->{a,b,c}_layout / op->{m,n} are consumed inside the
 * common emit helpers this phase forwards into).
 *
 * Peers (rocke_gfx1201_dfcp_build_ctx_init, the other closures) live in sibling
 * translation units and are reached via the shared internal header.
 */

#include <stddef.h>

#include "rocke/instance_gfx1201_deep_fused_conv_pool_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 * CLOSURE PHASE: epilogue_override(b, conv_spec_, accs, grid, y_rsrc, w1_rsrc)
 *
 * Walks (Python order, deep_fused_conv_pool.py:1334-1375):
 *   c_smem = _stage_accumulators_to_cshuffle_lds(b, op, accs, grid, sync=False)
 *   w1_smem = _load_conv1_weights_to_lds(b, spec, w1_rsrc, grid, sync=False)
 *   b.sync()                                   # single merged barrier
 *   defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue)   (staged ctx->defer)
 *   conv1_accs = _emit_conv1_1x1(..., defer_epilogue=defer)
 *   deferred_epi = spec.conv1_epilogue if defer else None
 *   if _maxpool_is_intra_lane(spec, grid):                      # MFMA fast path
 *       _emit_inline_maxpool_from_registers(..., epilogue=deferred_epi)
 *   elif _maxpool_is_intra_lane_wmma(spec, grid, op):           # WMMA fast path
 *       _emit_wmma_maxpool_from_registers(..., op, epilogue=deferred_epi)
 *   else:
 *       conv1_smem = _stage_accumulators_to_cshuffle_lds(b, op, conv1_accs, grid)
 *       _emit_inline_maxpool_from_cshuffle(..., epilogue=deferred_epi)
 * ===================================================================== */
void rocke_gfx1201_dfcp_epilogue_override(rocke_gfx1201_dfcp_build_ctx_t* ctx,
                                          const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                          rocke_value_t* const* accs,
                                          size_t num_accs,
                                          const rocke_warp_grid_t* grid,
                                          rocke_value_t* y_rsrc,
                                          rocke_value_t* w1_rsrc)
{
    rocke_ir_builder_t* b;
    const rocke_deep_fused_conv_pool_spec_t* spec; /* common spec view (&spec->base) */
    const rocke_mma_op_t* op; /* WMMA 16x16x16 op               */
    bool defer;
    const rocke_conv_acc_epilogue_t* deferred_epi;
    rocke_status_t st;
    size_t i;

    if(ctx == NULL || ctx->b == NULL)
    {
        return;
    }
    b = ctx->b;
    spec = ctx->common_spec; /* the family-agnostic emit helpers know only this */
    op = ctx->op; /* resolved WMMA op (wave32, m=n=k=16)             */

    /* Stage per-callback scratch onto the ctx so the body reads only the ctx. */
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->y_rsrc = y_rsrc;
    ctx->w1_rsrc = w1_rsrc;
    ctx->num_conv0_accs = 0;
    for(i = 0; i < num_accs && i < (size_t)ROCKE_GFX1201_DFCP_MAX_ACCS; ++i)
    {
        ctx->conv0_accs[i] = accs[i];
    }
    ctx->num_conv0_accs = (num_accs < (size_t)ROCKE_GFX1201_DFCP_MAX_ACCS)
                              ? num_accs
                              : (size_t)ROCKE_GFX1201_DFCP_MAX_ACCS;

    /* Barrier-merge: the conv0 cshuffle stage (writes DeepFusionC_smem) and the
     * W1 load (writes W1_smem) target disjoint LDS tiles, and the conv1 MMA below
     * reads both. Emit each producer without its own barrier and gate the
     * consumer on a single block-wide barrier; this also lets the W1 global loads
     * overlap the conv0 cshuffle LDS stores. */
    ctx->c_smem = rocke_dfcp_stage_accumulators_to_cshuffle_lds(
        b, op, accs, num_accs, grid, /*sync=*/false);
    ctx->w1_smem = rocke_dfcp_load_conv1_weights_to_lds(b,
                                                        spec,
                                                        w1_rsrc,
                                                        grid,
                                                        /*sync=*/false);
    rocke_b_sync(b);

    /* VALU opt: ReLU/bias/clamp/(scale>=0) are monotonic, so the conv1 epilogue
     * commutes with maxpool. Defer it past the pool to apply once per pooled
     * pixel instead of per conv1 acc element (~4x fewer fmax). The decision is
     * the one rocke_gfx1201_dfcp_build_ctx_init computed from spec.conv1_epilogue. */
    defer = ctx->defer;

    st = rocke_dfcp_emit_conv1_1x1(b,
                                   spec,
                                   conv_spec_,
                                   op,
                                   ctx->c_smem,
                                   ctx->w1_smem,
                                   grid,
                                   /*defer_epilogue=*/defer,
                                   ctx->conv1_accs,
                                   (size_t)ROCKE_GFX1201_DFCP_MAX_ACCS,
                                   &ctx->num_conv1_accs);
    if(st != ROCKE_OK)
    {
        /* error already routed through b by the emit helper */
        return;
    }

    deferred_epi = defer ? &spec->conv1_epilogue : NULL;
    ctx->deferred_epi = deferred_epi;

    if(rocke_dfcp_maxpool_is_intra_lane(spec, grid))
    {
        /* MFMA-32x32 register-resident fast path. Geometry-gated off for the WMMA
         * warp_tile 16x16, so this branch is never taken here; preserved to keep
         * the walk byte-faithful to the common closure. */
        rocke_dfcp_emit_inline_maxpool_from_registers(
            b, spec, ctx->conv1_accs, ctx->num_conv1_accs, y_rsrc, grid, deferred_epi);
    }
    else if(ctx->use_wmma_register_maxpool)
    {
        /* RDNA4 analogue: the 2x2 corners live in the same lane across the two
         * adjacent m-tile accs, so skip the cshuffle LDS handoff entirely. The
         * predicate (_maxpool_is_intra_lane_wmma(spec, grid, op)) was resolved
         * once into ctx->use_wmma_register_maxpool by the prologue. */
        rocke_dfcp_emit_wmma_maxpool_from_registers(
            b, spec, ctx->conv1_accs, ctx->num_conv1_accs, y_rsrc, grid, op, deferred_epi);
    }
    else
    {
        /* Generic layout-agnostic cshuffle-LDS gather + maxpool: stage the conv1
         * accs to LDS (with its own barrier) and reduce from there. */
        rocke_value_t* conv1_smem = rocke_dfcp_stage_accumulators_to_cshuffle_lds(
            b, op, ctx->conv1_accs, ctx->num_conv1_accs, grid, /*sync=*/true);
        ctx->conv1_smem = conv1_smem;
        rocke_dfcp_emit_inline_maxpool_from_cshuffle(
            b, spec, conv1_smem, y_rsrc, grid, deferred_epi);
    }
}
