// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_deep_fused_conv_pool_gfx950_dfcp_epilogue_override.c --
 * C99 part-file of the chunked port of the gfx950 (CDNA, wave64, MFMA 32x32x16)
 * arch shim over build_deep_fused_conv_pool
 *   (rocke/instances/gfx950/deep_fused_conv_pool.py -> re-exported common
 *    rocke/instances/common/deep_fused_conv_pool.py build_deep_fused_conv_pool,
 *    the `epilogue_override` nested closure, Python lines 1334-1375).
 *
 * SCOPE -- exactly ONE symbol:
 *   - rocke_gfx950_dfcp_epilogue_override          (epilogue_override closure)
 *
 * The fused write-back closure. Walks (byte-faithful Python order), driven by the
 * gfx950-resolved MFMA op (ctx->op, wave64, m=n=32, k=16):
 *
 *   c_smem  = _stage_accumulators_to_cshuffle_lds(b, op, accs, grid, sync=False)
 *   w1_smem = _load_conv1_weights_to_lds(b, spec, w1_rsrc, grid, sync=False)
 *   b.sync()                                   # single merged barrier
 *   defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue)  (staged ctx->defer)
 *   conv1_accs = _emit_conv1_1x1(..., defer_epilogue=defer)
 *   deferred_epi = spec.conv1_epilogue if defer else None
 *   if _maxpool_is_intra_lane(spec, grid):                     # gfx950 ACTIVE
 *       _emit_inline_maxpool_from_registers(..., epilogue=deferred_epi)
 *   else:
 *       conv1_smem = _stage_accumulators_to_cshuffle_lds(b, op, conv1_accs, grid)
 *       _emit_inline_maxpool_from_cshuffle(..., epilogue=deferred_epi)
 *
 * The conv0 cshuffle stage (writes DeepFusionC_smem) and the W1 load (writes
 * W1_smem) target disjoint LDS tiles and the conv1 MMA below reads both, so each
 * producer is emitted with sync=False and the consumer is gated on a single
 * merged block-wide barrier; this also overlaps the W1 global loads with the
 * conv0 cshuffle LDS stores.
 *
 * GFX950 MAXPOOL ROUTING. The MFMA-32x32 intra-lane register-resident fast path
 * is ACTIVE here, gated by ctx->use_intra_lane_maxpool
 * (_maxpool_is_intra_lane re-derived over the live per-callback grid). When the
 * geometry predicate is false the gfx950 path falls back to the generic
 * cshuffle-LDS gather maxpool (stage conv1 accs +
 * rocke_dfcp_emit_inline_maxpool_from_cshuffle). The WMMA register-resident maxpool
 * (rocke_dfcp_emit_wmma_maxpool_from_registers) is the gfx1201 counterpart's path
 * and is NOT reachable from this gfx950 shim.
 *
 * This TU reads/writes only ctx (conv0_accs/conv1_accs, c_smem/w1_smem/
 * conv1_smem, y_rsrc/w1_rsrc) and the builder it carries; peers are reached via
 * the internal header. No headers are edited here.
 */
#include "rocke/instance_gfx950_deep_fused_conv_pool_internal.h"

#include <stddef.h>

#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h"
#include "rocke/ir.h"

/* ===================================================================== *
 * CLOSURE PHASE: epilogue_override(b, conv_spec_, accs, grid, y_rsrc, w1_rsrc)
 * ===================================================================== */
void rocke_gfx950_dfcp_epilogue_override(rocke_gfx950_dfcp_build_ctx_t* ctx,
                                         const rocke_implicit_gemm_conv_spec_t* conv_spec_,
                                         rocke_value_t* const* accs,
                                         size_t num_accs,
                                         const rocke_warp_grid_t* grid,
                                         rocke_value_t* y_rsrc,
                                         rocke_value_t* w1_rsrc)
{
    rocke_ir_builder_t* b;
    const rocke_deep_fused_conv_pool_spec_t* spec; /* common spec view (&spec->base) */
    const rocke_mma_op_t* op; /* MFMA 32x32x16 op               */
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
    op = ctx->op; /* resolved MFMA op (wave64, m=n=32, k=16)         */

    /* Stage per-callback scratch onto the ctx so the body reads only the ctx. */
    ctx->conv_spec_cb = conv_spec_;
    ctx->grid = grid;
    ctx->y_rsrc = y_rsrc;
    ctx->w1_rsrc = w1_rsrc;
    ctx->num_conv0_accs = 0;
    for(i = 0; i < num_accs && i < (size_t)ROCKE_GFX950_DFCP_MAX_ACCS; ++i)
    {
        ctx->conv0_accs[i] = accs[i];
    }
    ctx->num_conv0_accs = (num_accs < (size_t)ROCKE_GFX950_DFCP_MAX_ACCS)
                              ? num_accs
                              : (size_t)ROCKE_GFX950_DFCP_MAX_ACCS;

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
     * pixel instead of per conv1 acc element (~4x fewer fmax). The decision is the
     * one rocke_gfx950_dfcp_build_ctx_init computed from spec.conv1_epilogue. */
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
                                   (size_t)ROCKE_GFX950_DFCP_MAX_ACCS,
                                   &ctx->num_conv1_accs);
    if(st != ROCKE_OK)
    {
        /* error already routed through b by the emit helper */
        return;
    }

    deferred_epi = defer ? &spec->conv1_epilogue : NULL;
    ctx->deferred_epi = deferred_epi;

    if(ctx->use_intra_lane_maxpool)
    {
        /* MFMA-32x32 register-resident fast path (gfx950 ACTIVE). Each lane's
         * vec<16> conv1 accumulator already holds the 4 pool windows it owns
         * (intra-lane, no shuffle), so reduce straight to global output. */
        rocke_dfcp_emit_inline_maxpool_from_registers(
            b, spec, ctx->conv1_accs, ctx->num_conv1_accs, y_rsrc, grid, deferred_epi);
    }
    else
    {
        /* Generic layout-agnostic cshuffle-LDS gather + maxpool: stage the conv1
         * accs to LDS (with its own barrier) and reduce from there. The WMMA
         * register-resident maxpool is the gfx1201 path and is not reachable
         * here. */
        rocke_value_t* conv1_smem = rocke_dfcp_stage_accumulators_to_cshuffle_lds(
            b, op, ctx->conv1_accs, ctx->num_conv1_accs, grid, /*sync=*/true);
        ctx->conv1_smem = conv1_smem;
        rocke_dfcp_emit_inline_maxpool_from_cshuffle(
            b, spec, conv1_smem, y_rsrc, grid, deferred_epi);
    }
}
